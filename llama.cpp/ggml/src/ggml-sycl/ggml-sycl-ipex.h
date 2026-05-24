#pragma once
// IPEX-LLM Fusion Kernel Integration — Three-Layer Architecture
//
// Layer 1 (dlopen):  Call IPEX-LLM .so functions directly
//   → Fastest path, requires ipex-binary/ at runtime
//   → Handles: Q4_K/Q5_K/Q6_K/Q4_0/Q8_0 dequant+GEMM
//
// Layer 2 (SPIR-V):  Load IPEX-LLM SPIR-V kernels via Level Zero interop
//   → No .so dependency, pure SYCL kernel execution
//   → Handles: QKV fusion, MLP fusion, SDP XMX attention
//
// Layer 3 (Graph):   Detect and rewrite computation graph patterns
//   → Replaces 3x matmul with 1x fused kernel at graph level
//   → Handles: Full attention block fusion, MoE expert fusion

#include <sycl/sycl.hpp>
#include <ggml.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

struct ggml_backend_sycl_context;  // forward decl from common.hpp

namespace ipex_fusion {

// ============================================================
// Common types
// ============================================================

using queue_ptr = std::shared_ptr<sycl::queue>;

enum class FusionLevel {
    NONE    = 0,
    DLOPEN  = 1,  // Use IPEX .so
    SPIRV   = 2,  // Use loaded SPIR-V kernel
    NATIVE  = 3,  // Use our own SYCL ESIMD kernel (future)
};

// ============================================================
// Layer 1: dlopen IPEX-LLM .so
// ============================================================

struct DlopenBackend {
    void *lib_handle = nullptr;

    // Per-token fused dequant+GEMM
    using VecMatFn = void (*)(const unsigned char *, const float *,
                              float *, int, int, sycl::queue &);
    VecMatFn vec_q4_0 = nullptr;
    VecMatFn vec_q4_K = nullptr;
    VecMatFn vec_q5_K = nullptr;
    VecMatFn vec_q6_K = nullptr;
    VecMatFn vec_q8_0 = nullptr;

    // Batched quantized GEMM (internal Q8_1 quant + SPIR-V kernel)
    using MulMatQFn = void (*)(ggml_backend_sycl_context &,
                               const ggml_tensor *, const ggml_tensor *,
                               ggml_tensor *, const char *, const float *,
                               const char *, float *, long, long, long, long,
                               sycl::queue *const &);
    MulMatQFn mul_mat_q = nullptr;

    bool load(const char *ipex_so_path);
    bool available() const { return lib_handle != nullptr && vec_q4_K != nullptr; }

    VecMatFn vec_mat_for(ggml_type t) const;
};

// ============================================================
// Layer 2: SPIR-V kernel loading via Level Zero
// ============================================================

struct SpirvKernel {
    std::string name;
    std::unique_ptr<sycl::kernel> sycl_kernel;
    int           num_args = 0;
};

struct SpirvBackend {
    struct Module {
        void        *ze_module = nullptr;  // ze_module_handle_t (opaque)
        std::unique_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> bundle;
        std::vector<SpirvKernel> kernels;
    };
    std::vector<Module> modules;

    bool load_spirv_file(const char *path, sycl::queue &q);
    bool load_recommended(sycl::queue &q);
    bool available() const { return !modules.empty(); }

    // Find a kernel by name substring
    const SpirvKernel *find_kernel(const std::string &name_substr) const;
};

// ============================================================
// Layer 3: Graph-level fusion detection
// ============================================================

// Describes a QKV pattern: 3 consecutive mul_mat ops projecting to Q, K, V
struct QKVPattern {
    const ggml_tensor *q_proj = nullptr;  // src0 of Q projection
    const ggml_tensor *k_proj = nullptr;  // src0 of K projection
    const ggml_tensor *v_proj = nullptr;  // src0 of V projection
    const ggml_tensor *input  = nullptr;  // shared src1 (activations)
    bool valid = false;
};

// Describes an MLP pattern: gate + up projections followed by down
struct MLPPattern {
    const ggml_tensor *gate_proj = nullptr;
    const ggml_tensor *up_proj   = nullptr;
    const ggml_tensor *down_proj = nullptr;
    const ggml_tensor *input     = nullptr;
    bool valid = false;
};

// Try to detect QKV pattern from 3 adjacent mul_mat ops
QKVPattern detect_qkv_pattern(const ggml_tensor *q_tensor,
                               const ggml_tensor *k_tensor,
                               const ggml_tensor *v_tensor);

// Try to detect MLP pattern from gate/up/down mul_mat ops
MLPPattern detect_mlp_pattern(const ggml_tensor *gate_tensor,
                               const ggml_tensor *up_tensor,
                               const ggml_tensor *down_tensor);

// ============================================================
// Unified dispatch — called from ggml_sycl_op_mul_mat
// ============================================================

struct FusionContext {
    DlopenBackend dlopen;
    SpirvBackend  spirv;
    bool          initialized       = false;
    bool          spirv_attempted   = false;

    void init();
};

// Global instance
FusionContext &get_fusion_ctx();

// Main entry point: try to handle a mul_mat via fusion kernel
// Returns true if handled, false if caller should fall through to oneMKL path.
bool try_fused_mul_mat(
    const ggml_tensor *src0, const ggml_tensor *src1,
    ggml_tensor *dst,
    const float *src1_fp32, float *dst_fp32,
    int64_t M, int64_t N, int64_t K,
    sycl::queue &q,
    ggml_backend_sycl_context *backend_ctx = nullptr,
    const char *src0_dd = nullptr,
    const char *dst_dd = nullptr,
    long row_low = 0, long row_high = 0,
    long padded_row_size = 0,
    FusionLevel min_level = FusionLevel::DLOPEN);

} // namespace ipex_fusion
