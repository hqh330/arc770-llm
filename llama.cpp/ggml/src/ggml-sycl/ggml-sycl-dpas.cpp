// s8 DPAS GEMM kernel: native XMX matrix multiply for Q8_0 quantized weights.
// Reference: reference/sycl-dpas/dpas.hpp, reference/sycl-esimd/dpas_common.hpp (read-only).
// Pattern: Q.single_task([=]() SYCL_ESIMD_KERNEL { ... }) from dpas_common.hpp:342
//
// Integration: insert in ggml-sycl.cpp, bypassing Q4_K→FP16 CPU conversion.
// Weights stay int8 → XMX directly. Expected +30-40% throughput vs current FP16 path.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;

// Host-side function: submit ESIMD DPAS kernel to SYCL queue.
// M: output rows (weight rows), N: output cols (activation cols), K: inner dim
// weights_s8:  Q8_0 quantized weights [M][K]  (int8)
// activations: pre-quantized inputs [K][N] (uint8, VNNI-packed by caller)
// output:      result [M][N] (float32)
// scale:       dequantization factor
void ggml_sycl_mul_mat_q8_xmx(
    sycl::queue &q,
    const uint8_t *weights_s8,    // [M][K]
    const uint8_t *activations,   // [K][N] VNNI-packed
    float *output,                // [M][N]
    int M, int N, int K,
    float scale
) {
    constexpr int SD = 8;   // SystolicDepth (fixed by hardware)
    constexpr int RC = 4;   // RepeatCount (M tile, 1-8)
    constexpr int ES = 8;   // ExecutionSize (N tile, 8 or 16)
    constexpr int TK = 32;  // K tile: SD * OpsPerChannel = 8*4 = 32 for s8

    // Allocate device memory with alignment for ESIMD
    auto *d_w = sycl::aligned_alloc_device<uint8_t>(128, M * K, q);
    auto *d_a = sycl::aligned_alloc_device<uint8_t>(128, K * N, q);
    auto *d_o = sycl::aligned_alloc_device<float>(128, M * N, q);

    q.memcpy(d_w, weights_s8, M * K);
    q.memcpy(d_a, activations, K * N);
    q.wait();

    q.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() SYCL_ESIMD_KERNEL {
            // Outer tile loops
            for (int m = 0; m < M; m += RC) {
            for (int n = 0; n < N; n += ES) {

                simd<int, ES> c_acc = 0;    // int32 accumulator

                // K loop: accumulate over K dimension in TK-sized tiles
                for (int k = 0; k < K; k += TK) {

                    // Load weight tile: [RC][TK] s8 → simd register
                    constexpr int AN = RC * TK / sizeof(uint8_t); // 4*32 = 128
                    simd<uint8_t, AN> a_tile(
                        (uint8_t*)(d_w + m * K + k),
                        overaligned_tag<16>{});

                    // Load activation tile: [TK][ES] u8 → simd register
                    // Activation must be VNNI-packed: [TK/2][ES][2]
                    constexpr int BN = TK * ES / sizeof(uint8_t); // 32*8 = 256
                    simd<uint8_t, BN> b_tile(
                        (uint8_t*)(d_a + k * N + n),
                        overaligned_tag<16>{});

                    // XMX DPAS: C[RC×ES] += A[RC×TK] × B[TK×ES]
                    // One hardware instruction, 512 ops per cycle per XMX engine
                    c_acc = dpas<SD, RC, int, int, uint8_t, uint8_t,
                        dpas_argument_type::u8, dpas_argument_type::s8>(
                        c_acc, b_tile, a_tile);
                }

                // Dequantize: int32 → fp32, apply scale, store
                simd<float, ES> c_fp32 = c_acc * scale;

                uint8_t *dst = (uint8_t*)(d_o + m * N + n);
                simd<float, ES>(c_fp32).copy_to((float*)dst);
            }}
        });
    }).wait();

    q.memcpy(output, d_o, M * N * sizeof(float)).wait();
    sycl::free(d_w, q);
    sycl::free(d_a, q);
    sycl::free(d_o, q);
}

// Integration into ggml-sycl.cpp (pseudocode):
//
// In ggml_sycl_op_mul_mat_sycl(), after the GGML_SYCL_F16 check:
//
// #ifdef GGML_SYCL_USE_DPAS  // new CMake flag
//   if (ggml_is_quantized(src0->type) && row_diff == src0->ne[1]) {
//       // Quantize activations to uint8 (one-time, on GPU)
//       ggml_sycl_mul_mat_q8_xmx(
//           *stream,
//           (const uint8_t*)src0_dd_i,  // Q8_0 weights (already int8 on device)
//           activations_u8,             // pre-quantized activations
//           dst_dd_i,
//           row_diff, src1_ncols, ne10,
//           ggml_fp16_to_fp32(scale));
//       return;
//   }
// #endif
