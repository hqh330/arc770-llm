// s4 DPAS GEMM kernel: Q4_0 weights directly on GPU via XMX (no CPU requantization).
// Reference: reference/sycl-dpas/dpas.hpp (read-only).
// Builds on: work/phase2-dpas-kernel/ggml-sycl-dpas.cpp (s8 DPAS framework).
// Requires: work/phase0/repack-vnni.cpp (VNNI weight repack, run once at model load).
//
// Key difference from s8 DPAS:
//   s8: RC=4, TK=32 → C[4×8] += A[4×32] × B[32×8]
//   s4: RC=8, TK=64 → C[8×8] += A[8×64] × B[64×8]
//   Same byte count per tile, but double the effective K elements (4-bit density).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdint>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;

void ggml_sycl_mul_mat_q4_xmx(
    sycl::queue &q,
    const uint8_t *weights_q4_vnni,  // Q4_0 weights, VNNI-repacked [M][K/2]
    const uint8_t *activations,      // uint8 activations [K][N] VNNI-packed
    float *output,                   // [M][N]
    int M, int N, int K,
    float scale
) {
    constexpr int SD = 8;   // SystolicDepth (fixed)
    constexpr int RC = 8;   // RepeatCount (M tile, 8 for s4 — double s8's 4)
    constexpr int ES = 8;   // ExecutionSize (N tile)
    constexpr int TK = 64;  // K tile: SD * OpsPerChannel = 8*8 = 64 for s4 (double s8's 32)

    int K_bytes = K / 2;    // 4-bit: K elements → K/2 bytes (VNNI still uses byte-aligned)

    auto *d_w = sycl::aligned_alloc_device<uint8_t>(128, M * K_bytes, q);
    auto *d_a = sycl::aligned_alloc_device<uint8_t>(128, K * N, q);
    auto *d_o = sycl::aligned_alloc_device<float>(128, M * N, q);

    q.memcpy(d_w, weights_q4_vnni, M * K_bytes);
    q.memcpy(d_a, activations, K * N);
    q.wait();

    q.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() SYCL_ESIMD_KERNEL {
            for (int m = 0; m < M; m += RC) {
            for (int n = 0; n < N; n += ES) {

                simd<int, ES> c_acc = 0;

                for (int kb = 0; kb < K_bytes; kb += (TK / 2)) { // TK/2 bytes for s4

                    // Weight tile: [RC][TK_bytes/2] = [8][32] s4 VNNI → register
                    constexpr int AN_s4 = RC * TK / 2 / sizeof(uint8_t); // 8*32 = 256
                    simd<uint8_t, AN_s4> a_tile(
                        (uint8_t*)(d_w + m * K_bytes + kb),
                        overaligned_tag<16>{});

                    // Activation tile: [TK][ES] u8
                    constexpr int BN_s4 = TK * ES / sizeof(uint8_t); // 64*8 = 512
                    simd<uint8_t, BN_s4> b_tile(
                        (uint8_t*)(d_a + kb * 2 * N + n),
                        overaligned_tag<16>{});

                    // XMX s4 DPAS: C[8×8] += A_s4[8×64] × B_u8[64×8]
                    c_acc = dpas<SD, RC, int, int, uint8_t, uint8_t,
                        dpas_argument_type::u4, dpas_argument_type::s4>(
                        c_acc, b_tile, a_tile);
                }

                simd<float, ES> c_fp32 = c_acc * scale;
                ((simd<float, ES>)c_fp32).copy_to(d_o + m * N + n);
            }}
        });
    }).wait();

    q.memcpy(output, d_o, M * N * sizeof(float)).wait();
    sycl::free(d_w, q); sycl::free(d_a, q); sycl::free(d_o, q);
}
