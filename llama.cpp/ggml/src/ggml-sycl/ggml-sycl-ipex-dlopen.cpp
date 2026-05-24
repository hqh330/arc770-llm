// Layer 1: dlopen IPEX-LLM libggml-sycl.so
// Calls IPEX-LLM's dequantize+GEMM fused kernels directly from our SYCL queue.
// Zero new GPU code — just host-side glue resolving symbols and forwarding calls.
//
// IPEX-LLM .so is expected at: ../ipex-binary/libggml-sycl.so
// (relative to llama.cpp build dir, or set IPEX_SO_PATH env var)

#include "ggml-sycl-ipex.h"
#include "common.hpp"
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ggml-impl.h>

namespace ipex_fusion {

// ============================================================
// DlopenBackend
// ============================================================

bool DlopenBackend::load(const char *ipex_so_path) {
    if (lib_handle) return true;

    if (!ipex_so_path)
        ipex_so_path = std::getenv("IPEX_SO_PATH");
    if (!ipex_so_path)
        ipex_so_path = "../ipex-binary/libggml-sycl.so";

    lib_handle = dlopen(ipex_so_path, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!lib_handle) {
        fprintf(stderr, "[IPEX-dlopen] Cannot load %s: %s\n", ipex_so_path, dlerror());
        return false;
    }

    // Per-token dequant+GEMM vec kernels (fallback)
    vec_q4_0 = (VecMatFn)dlsym(lib_handle,
        "_Z41ggml_sycl_op_dequantize_mul_mat_vec_q4_0PKhPKfPfiiRN4sycl3_V15queueE");
    vec_q4_K = (VecMatFn)dlsym(lib_handle,
        "_Z40ggml_sycl_op_dequantize_mul_mat_vec_q4_KPKhPKfPfiiRN4sycl3_V15queueE");
    vec_q5_K = (VecMatFn)dlsym(lib_handle,
        "_Z40ggml_sycl_op_dequantize_mul_mat_vec_q5_KPKhPKfPfiiRN4sycl3_V15queueE");
    vec_q6_K = (VecMatFn)dlsym(lib_handle,
        "_Z40ggml_sycl_op_dequantize_mul_mat_vec_q6_kPKhPKfPfiiRN4sycl3_V15queueE");
    vec_q8_0 = (VecMatFn)dlsym(lib_handle,
        "_Z41ggml_sycl_op_dequantize_mul_mat_vec_q8_0PKhPKfPfiiRN4sycl3_V15queueE");

    // Batched quantized GEMM (quantizes activations to Q8_1 internally,
    // then launches SPIR-V kernel for full dequant+GEMM)
    mul_mat_q = (MulMatQFn)dlsym(lib_handle,
        "_Z22ggml_sycl_op_mul_mat_qR25ggml_backend_sycl_contextPK11ggml_tensorS3_PS1_PKcPKfS6_PfllllRKPN4sycl3_V15queueE");

    fprintf(stderr, "[IPEX-dlopen] Loaded vec kernels: q4_K=%p q5_K=%p mul_mat_q=%p\n",
            (void*)vec_q4_K, (void*)vec_q5_K, (void*)mul_mat_q);

    return vec_q4_K != nullptr;
}

DlopenBackend::VecMatFn DlopenBackend::vec_mat_for(ggml_type t) const {
    switch (t) {
        case GGML_TYPE_Q4_0: return vec_q4_0;
        case GGML_TYPE_Q4_K: return vec_q4_K;
        case GGML_TYPE_Q5_K: return vec_q5_K;
        case GGML_TYPE_Q6_K: return vec_q6_K;
        case GGML_TYPE_Q8_0: return vec_q8_0;
        default: return nullptr;
    }
}

// ============================================================
// FusionContext
// ============================================================

void FusionContext::init() {
    if (initialized) return;
    initialized = true;
    dlopen.load(nullptr);
}

FusionContext &get_fusion_ctx() {
    static FusionContext ctx;
    return ctx;
}

// ============================================================
// Main dispatch: try_fused_mul_mat
// ============================================================

bool try_fused_mul_mat(
    const ggml_tensor *src0, const ggml_tensor *src1,
    ggml_tensor *dst,
    const float *src1_fp32, float *dst_fp32,
    int64_t M, int64_t N, int64_t K,
    sycl::queue &q,
    ggml_backend_sycl_context *backend_ctx,
    const char *src0_dd,
    const char *dst_dd,
    long row_low, long row_high,
    long padded_row_size,
    FusionLevel min_level)
{
    auto &ctx = get_fusion_ctx();
    if (!ctx.initialized) ctx.init();

    // Lazy-load SPIR-V kernels on first call
    if (!ctx.spirv_attempted) {
        ctx.spirv_attempted = true;
        if (q.get_device().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            fprintf(stderr, "[IPEX-spirv] Loading recommended kernels...\n");
            ctx.spirv.load_recommended(q);
        }
    }

    ggml_type t = src0->type;

    // --- Layer 1: dlopen per-token vec-mat (opt-in via IPEX_FORCE=1) ---
    // The per-token dlopen overhead currently makes this slower than MKL
    // for both prompt and generation. Enable only for testing.
    static int ipex_force = []() {
        const char *v = std::getenv("IPEX_FORCE");
        return v ? std::atoi(v) : 0;
    }();

    if (min_level <= FusionLevel::DLOPEN && ctx.dlopen.available() && ipex_force) {
        auto vec_fn = ctx.dlopen.vec_mat_for(t);

        if (vec_fn) {
            const auto *d_weights = (const unsigned char *)src0->data;

            for (int64_t tok = 0; tok < N; tok++) {
                vec_fn(d_weights,
                       src1_fp32 + tok * K,
                       dst_fp32  + tok * M,
                       (int)K, (int)M, q);
            }
            q.wait();
            return true;
        }
    }

    // --- Layer 2: SPIR-V path ---
    if (min_level <= FusionLevel::SPIRV && ctx.spirv.available()) {
        const SpirvKernel *kern = nullptr;
        switch (t) {
            case GGML_TYPE_Q4_K:
                kern = ctx.spirv.find_kernel("ggml_mul_mat_q4_K_q8_1_sycl");
                break;
            case GGML_TYPE_Q4_0:
                kern = ctx.spirv.find_kernel("linear_forward_kernel");
                break;
            default: break;
        }
        if (kern && kern->sycl_kernel) {
            fprintf(stderr, "[IPEX-spirv] Found kernel '%s' for type %d"
                    " (args=%d) — dispatch pending\n",
                    kern->name.c_str(), (int)t, kern->num_args);
        }
    }

    return false;
}

} // namespace ipex_fusion
