// Layer 2: SPIR-V kernel loading via Level Zero interop
// Loads IPEX-LLM's fusion kernel SPIR-V modules and wraps them as SYCL kernels.
//
// Requires:
//   - Level Zero loader (libze_loader)
//   - SPIR-V files at ../ipex-kernels/kernel_*.spv
//   - oneAPI 2025.x with SYCL Level Zero backend

#include "ggml-sycl-ipex.h"

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>

namespace ipex_fusion {

// ============================================================
// SpirvBackend
// ============================================================

bool SpirvBackend::load_spirv_file(const char *path, sycl::queue &q) {
    // 1. Read SPIR-V binary from file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[IPEX-spirv] Cannot open: %s\n", path);
        return false;
    }
    std::vector<uint8_t> spirv_bin(std::istreambuf_iterator<char>(file), {});

    // 2. Get native Level Zero device and context from SYCL
    auto ze_device  = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        q.get_device());
    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        q.get_context());

    // 3. Create Level Zero module from SPIR-V IL
    ze_module_desc_t mod_desc = {};
    mod_desc.stype  = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mod_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    mod_desc.inputSize  = spirv_bin.size();
    mod_desc.pInputModule = spirv_bin.data();

    ze_module_handle_t ze_mod = nullptr;
    ze_module_build_log_handle_t build_log = nullptr;
    ze_result_t zr = zeModuleCreate(ze_context, ze_device, &mod_desc,
                                     &ze_mod, &build_log);
    if (zr != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[IPEX-spirv] zeModuleCreate failed: 0x%x for %s\n",
                zr, path);
        if (build_log) {
            size_t log_size = 0;
            zeModuleBuildLogGetString(build_log, &log_size, nullptr);
            std::vector<char> log(log_size + 1);
            zeModuleBuildLogGetString(build_log, &log_size, log.data());
            fprintf(stderr, "[IPEX-spirv] Build log: %s\n", log.data());
            zeModuleBuildLogDestroy(build_log);
        }
        return false;
    }

    // 4. Wrap module as SYCL kernel_bundle
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero,
        sycl::kernel_bundle<sycl::bundle_state::executable>> bundle_in{
            ze_mod, sycl::ext::oneapi::level_zero::ownership::keep};

    auto bundle = sycl::make_kernel_bundle<
        sycl::backend::ext_oneapi_level_zero,
        sycl::bundle_state::executable>(bundle_in, q.get_context());

    // 5. Enumerate kernels in the module
    uint32_t name_count = 0;
    zeModuleGetKernelNames(ze_mod, &name_count, nullptr);
    std::vector<const char *> knames(name_count);
    zeModuleGetKernelNames(ze_mod, &name_count, knames.data());

    Module mod;
    mod.ze_module = ze_mod;
    mod.bundle    = bundle;

    for (uint32_t i = 0; i < name_count; i++) {
        std::string name(knames[i]);

        // Skip Intel debug symbol table entries
        if (name.find("Intel_Symbol") != std::string::npos) continue;

        // Create Level Zero kernel
        ze_kernel_desc_t kdesc = {};
        kdesc.stype      = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        kdesc.pKernelName = knames[i];
        ze_kernel_handle_t ze_kern = nullptr;
        zr = zeKernelCreate(ze_mod, &kdesc, &ze_kern);
        if (zr != ZE_RESULT_SUCCESS) continue;

        // Wrap as SYCL kernel
        sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero,
            sycl::kernel> kern_in{mod.bundle, ze_kern,
                sycl::ext::oneapi::level_zero::ownership::keep};
        auto sk = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
            kern_in, q.get_context());

        SpirvKernel k;
        k.name     = name;
        k.sycl_kernel = sk;
        k.num_args = sk.get_info<sycl::info::kernel::num_args>();

        mod.kernels.push_back(std::move(k));
        zeKernelDestroy(ze_kern);
    }

    fprintf(stderr, "[IPEX-spirv] Loaded %zu kernels from %s\n",
            mod.kernels.size(), path);

    modules.push_back(std::move(mod));
    return true;
}

const SpirvKernel *SpirvBackend::find_kernel(
    const std::string &name_substr) const
{
    for (auto &mod : modules) {
        for (auto &k : mod.kernels) {
            if (k.name.find(name_substr) != std::string::npos)
                return &k;
        }
    }
    return nullptr;
}

// ============================================================
// Predefined kernel launch helpers (Layer 2 dispatch)
// ============================================================

// Map of: "which kernel to load" → "which models benefit"
//
// kernel_0.spv  → linear_forward_kernel (Q4_0 dequant+GEMM)
//                → Used for Q4_0 models, replaces host-side dequant
//
// kernel_6.spv  → qlinear_xpu_kernel_q4_0_2x16_qkv (QKV fusion)
//                → Used for attention Q/K/V projection fusion
//                → Replaces 3 separate GEMM calls with 1
//
// kernel_9.spv  → mlp_forward_q4_k_kernel (MLP Q4_K fusion)
//                → Used for FFN gate+up projection fusion
//
// kernel_3.spv  → sdp_causal_xmx_kernel (SDP attention with DPAS/XMX)
//                → Used for the attention score computation itself
//                → This is the XMX-accelerated flash attention!
//
// kernel_14.spv → vec_q4_K_batch_kernel (Q4_K batch dequant+GEMM)
//                → Alternative to dlopen batch_forward_q4_K
//                → Pure SPIR-V, no .so dependency
//
// kernel_24.spv → ggml_mul_mat_q4_K_q8_1_sycl (Q4_K × Q8_1 GEMM)
//                → Activations must be quantized to Q8_1 first
//
// kernel_12/43.spv → KV cache quantize/dequant

// Recommended loading order:
//   1. kernel_0  (Q4_0 → immediate win for Q4_0 models)
//   2. kernel_24 (Q4_K GEMM → biggest win for Q4_K models like Qwen 9B)
//   3. kernel_6  (QKV fusion → attention optimization)
//   4. kernel_9  (MLP fusion → FFN optimization)
//   5. kernel_3  (SDP XMX → DPAS attention)

bool SpirvBackend::load_recommended(sycl::queue &q) {
    const char *kernel_dir = std::getenv("IPEX_KERNEL_DIR");
    if (!kernel_dir) kernel_dir = "../ipex-kernels";

    char path[512];
    const char *files[] = {
        "kernel_0.spv",   // Q4_0 linear_forward
        "kernel_24.spv",  // Q4_K mul_mat
        "kernel_6.spv",   // QKV fusion
        "kernel_9.spv",   // MLP fusion
        "kernel_3.spv",   // SDP XMX
    };

    int loaded = 0;
    for (auto f : files) {
        snprintf(path, sizeof(path), "%s/%s", kernel_dir, f);
        if (load_spirv_file(path, q))
            loaded++;
    }
    return loaded > 0;
}

} // namespace ipex_fusion
