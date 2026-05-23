// Layer 1: dlopen IPEX-LLM libggml-sycl.so
// Calls IPEX-LLM's format converters + batch_forward_q4_K directly.
// This is the fastest path — zero new GPU code, just host-side glue.
//
// IPEX-LLM .so is expected at: ../ipex-binary/libggml-sycl.so
// (relative to llama.cpp build dir, or set IPEX_SO_PATH env var)

#include "ggml-sycl-ipex.h"
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
    if (lib_handle) return true;  // already loaded

    if (!ipex_so_path)
        ipex_so_path = std::getenv("IPEX_SO_PATH");
    if (!ipex_so_path)
        ipex_so_path = "../ipex-binary/libggml-sycl.so";

    lib_handle = dlopen(ipex_so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib_handle) {
        fprintf(stderr, "[IPEX-dlopen] Cannot load %s: %s\n", ipex_so_path, dlerror());
        return false;
    }

    // Resolve format converters (C++ mangled names from IPEX-LLM 2.3.0)
    q4_0_convert = (ConvertFn)dlsym(lib_handle,
        "_Z31ggml_q4_0_format_convert_to_xpuPKvPvm");
    q4_K_convert = (ConvertFn)dlsym(lib_handle,
        "_Z31ggml_q4_K_format_convert_to_xpuPKvPvm");
    q5_K_convert = (ConvertFn)dlsym(lib_handle,
        "_Z31ggml_q5_K_format_convert_to_xpuPKvPvm");
    q6_K_convert = (ConvertFn)dlsym(lib_handle,
        "_Z31ggml_q6_K_format_convert_to_xpuPKvPvm");
    q8_0_convert = (ConvertFn)dlsym(lib_handle,
        "_Z31ggml_q8_0_format_convert_to_xpuPKvPvm");

    // Resolve batch forward functions
    batch_q4k = (BatchFn)dlsym(lib_handle,
        "_Z18batch_forward_q4_KPKfPKhPflllRN4sycl3_V15queueE");

    fprintf(stderr, "[IPEX-dlopen] Loaded: q4_K=%p q5_K=%p q6_K=%p batch_q4k=%p\n",
            (void*)q4_K_convert, (void*)q5_K_convert,
            (void*)q6_K_convert, (void*)batch_q4k);

    return batch_q4k != nullptr;
}

DlopenBackend::ConvertFn DlopenBackend::converter_for(ggml_type t) const {
    switch (t) {
        case GGML_TYPE_Q4_0: return q4_0_convert;
        case GGML_TYPE_Q4_K: return q4_K_convert;
        case GGML_TYPE_Q5_K: return q5_K_convert;
        case GGML_TYPE_Q6_K: return q6_K_convert;
        case GGML_TYPE_Q8_0: return q8_0_convert;
        default: return nullptr;
    }
}

DlopenBackend::BatchFn DlopenBackend::batch_for(ggml_type t) const {
    // batch_forward_q4_K handles Q4_K, Q5_K, Q6_K
    // (IPEX-LLM's internal dispatch routes to correct kernel variant)
    switch (t) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return batch_q4k;
        default:
            return nullptr;
    }
}

const uint8_t *DlopenBackend::get_converted_weights(
    const ggml_tensor *tensor, size_t &out_size)
{
    const void *key = tensor->data;
    size_t src_size = ggml_nbytes(tensor);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto &cached = weight_cache[key];
        if (!cached.empty()) {
            out_size = cached.size();
            return cached.data();
        }
        // Allocate 2x for IPEX format expansion, convert, cache
        cached.resize(src_size * 2);
        ConvertFn conv = converter_for(tensor->type);
        if (!conv) {
            fprintf(stderr, "[IPEX-dlopen] No converter for type %d\n", tensor->type);
            return nullptr;
        }
        conv(tensor->data, cached.data(), src_size);
        out_size = cached.size();
        fprintf(stderr, "[IPEX-dlopen] Converted weights: %zu → %zu bytes (type=%d)\n",
                src_size, out_size, tensor->type);
        return cached.data();
    }
}

// ============================================================
// FusionContext
// ============================================================

void FusionContext::init() {
    if (initialized) return;
    initialized = true;

    // Layer 1: try dlopen
    dlopen.load(nullptr);  // uses IPEX_SO_PATH or default path

    // Layer 2: try SPIR-V loading (deferred — requires active SYCL queue)
    // spirv.load_spirv_file(...) is called lazily on first use
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
    const float *src1_fp32, float *dst_fp32,
    int64_t M, int64_t N, int64_t K,
    sycl::queue &q,
    FusionLevel min_level)
{
    auto &ctx = get_fusion_ctx();
    if (!ctx.initialized) ctx.init();

    ggml_type t = src0->type;

    // --- Layer 1: dlopen path ---
    if (min_level <= FusionLevel::DLOPEN && ctx.dlopen.available()) {
        auto converter = ctx.dlopen.converter_for(t);
        auto batch_fn  = ctx.dlopen.batch_for(t);

        if (converter && batch_fn) {
            // 1. Get IPEX-format weights (converted + cached)
            size_t ipex_wsize = 0;
            const uint8_t *ipex_weights = ctx.dlopen.get_converted_weights(
                src0, ipex_wsize);
            if (!ipex_weights) return false;

            // 2. Upload converted weights to device
            //    (caller must ensure src1_fp32 and dst_fp32 are already on device)
            auto *d_weights = sycl::malloc_device<uint8_t>(ipex_wsize, q);
            if (!d_weights) return false;
            q.memcpy(d_weights, ipex_weights, ipex_wsize);
            q.wait();

            // 3. Launch IPEX-LLM batch forward
            //    batch_forward_q4_K(input_fp32, weights_ipex, output_fp32, M, N, K, queue)
            batch_fn(src1_fp32, d_weights, dst_fp32, M, N, K, q);
            q.wait();

            sycl::free(d_weights, q);
            return true;
        }
    }

    // --- Layer 2: SPIR-V path ---
    if (min_level <= FusionLevel::SPIRV && ctx.spirv.available()) {
        // Look for a kernel matching this quant type + problem size
        const SpirvKernel *kern = nullptr;

        switch (t) {
            case GGML_TYPE_Q4_K:
                // kernel_24.spv: ggml_mul_mat_q4_K_q8_1_sycl
                // Requires activations quantized to Q8_1 first
                kern = ctx.spirv.find_kernel("ggml_mul_mat_q4_K_q8_1_sycl");
                break;
            case GGML_TYPE_Q4_0:
                // kernel_0.spv: linear_forward_kernel (FP32 in → FP32 out)
                kern = ctx.spirv.find_kernel("linear_forward_kernel");
                break;
            default:
                break;
        }

        if (kern) {
            // Launch via SYCL interop kernel
            // (Detailed arg binding depends on kernel — see kernel-signatures.md)
            // TODO: implement per-kernel arg binding
            (void)kern;  // placeholder
        }
    }

    // --- Layer 3: Native SYCL ESIMD (future) ---
    // TODO: our own dequant+GEMM fusion kernel

    return false;
}

} // namespace ipex_fusion
