# IPEX-LLM Fusion Kernel Signatures (Reverse-Engineered)

Extracted from IPEX-LLM 2.3.0b20250724 `libggml-sycl.so` via `nm -D` + `c++filt`.

## Q4_0 Dequant+GEMM (kernel_0.spv)

```cpp
// linear_forward_kernel<float, 2, 4, 16, 11>
// (const uint8_t* weights_q4_0, const float* input_fp32,
//  float* output_fp32, int M, int N, sycl::queue&)
```

## Q4_K Batch GEMM (kernel_14.spv)

```cpp
// vec_q4_K_batch_kernel<float, 2, 1, 16, 8, 64, true/false, false>
// (const void* input, const uint8_t* weights_q4k,
//  const uint8_t* scales, void* output,
//  int M, int N, int K, sycl::queue&)
```

## QKV Fusion (kernel_6.spv)

```cpp
// qlinear_xpu_kernel_q4_0_2x16_qkv<64, 32, 2>
// (const float* input,
//  const uint8_t* w_q, const uint8_t* w_k, const uint8_t* w_v,
//  const float* bias_q, const float* bias_k, const float* bias_v,
//  const int* neox_info,
//  half* q_out, half* k_out, half* v_out,
//  unsigned long d0..d6, float scale_q, float scale_k, float scale_v,
//  sycl::queue&)

// WQKV Fusion (combined weight):
// qlinear_xpu_kernel_q4_0_2x16_wqkv_neox<64, 32, 2>
// (const float* input, const uint8_t* w_combined, const float* bias,
//  const int* neox_info,
//  half* q_out, half* k_out, half* v_out,
//  unsigned long d0..d6, float scale_q, float scale_k, float scale_v,
//  sycl::queue&)
```

## MLP Fusion (kernel_9.spv)

```cpp
// mlp_forward_q4_k_kernel<float, 2, 4, 32, 12>
// (const float* input, float* output,
//  const uint8_t* w_gate, const uint8_t* w_up,
//  const float* bias_gate, const float* bias_up,
//  unsigned long n_embd, unsigned long n_ff, unsigned long batch,
//  int, sycl::queue&)
```

## SDP XMX Attention (kernel_3.spv)

```cpp
// sdp_causal_xmx_kernel<128, 64, 8, 16, 16>
// (const void* Q, const void* K, const void* V,
//  const void* mask, const void*, const void*,
//  float* output, long dims[19],
//  int, int, int, int, int, float scale, sycl::queue&)
```

## KV Cache Quantization

```cpp
// ggml_sycl_op_quantize_kv
// (const half* k_in, const half* v_in,
//  uint8_t* k_out, uint8_t* v_out,
//  unsigned long n_embd_k, unsigned long n_tokens,
//  sycl::queue&)

// ggml_sycl_op_dequantize_kv
// (const uint8_t* k_in, half* k_out,
//  bool has_scale, unsigned long dims[7],
//  sycl::queue&)
```

## Host-Side Wrappers (callable via dlopen)

```cpp
// Format converters:
void ggml_q4_0_format_convert_to_xpu(const void* src, void* dst, unsigned long size);
void ggml_q4_K_format_convert_to_xpu(const void* src, void* dst, unsigned long size);
void ggml_q5_K_format_convert_to_xpu(const void* src, void* dst, unsigned long size);
void ggml_q6_K_format_convert_to_xpu(const void* src, void* dst, unsigned long size);
void ggml_q8_0_format_convert_to_xpu(const void* src, void* dst, unsigned long size);
void ggml_q4_0_format_convert_to_xpu_ipex_llm_blksize(const void* src, void* dst, unsigned long size);

// Batch forward (GPU fused dequant+GEMM):
void batch_forward_q4_K(const float* input, const unsigned char* weights,
                        float* output, long M, long N, long K, sycl::queue&);
```

## SPIR-V Module Map

| File | Content | Key Kernels |
|------|---------|-------------|
| kernel_0.spv | Q4_0 linear forward | linear_forward_kernel ×6 |
| kernel_3.spv | SDP causal XMX | sdp_causal_xmx_kernel ×4 |
| kernel_4.spv | SDP FP8 causal XMX | sdp_fp8_causal_xmx_kernel |
| kernel_6.spv | QKV/WQKV fusion | qlinear_xpu_kernel_q4_0_2x16_qkv ×3 |
| kernel_9.spv | MLP Q4_K fusion | mlp_forward_q4_k_kernel |
| kernel_10-11.spv | MLP Q4_0 fusion | mlp_forward_q4_0_kernel |
| kernel_14.spv | vec Q4_K batch | vec_q4_K_batch_kernel ×80 |
| kernel_23.spv | mul_mat_vec Q4_K | mul_mat_vec_q4_K_q8_1_sycl |
| kernel_24.spv | mul_mat Q4_K/Q6_K/Q8_0 | ggml_mul_mat_q4_K_q8_1_sycl |
| kernel_28.spv | Q4_K weight reorder | reorder_qw_q4_k |
| kernel_43.spv | Q4_K dequant row | dequantize_new_row_q4_K_sycl |
