# BigDL SYCL ESIMD Kernel 参考

来源: `/home/w/llama.cpp项目扩展/bigDL/BigDL-core-main/BigDL-core-main/bigdl-core-xe/`

## 文件说明

| 文件 | 内容 | 参考价值 |
|------|------|----------|
| `norm.cpp` | RMSNorm + LayerNorm SYCL ESIMD kernel 完整实现 | 自研 kernel 模板 |
| `norm.h` | 函数声明 | — |
| `utils.h` | sycl::queue 获取 + submit_kernel 封装 | 队列管理模板 |
| `quantize.c` / `quantize.h` | Q4_0 GPTQ CPU 量化参考实现 | 量化算法参考 |

## 核心模式 (来自 norm.cpp)

```
1. slm_init<size>()                        — 分配共享本地内存
2. block_load<T,N>(ptr)                    — 从全局内存块读取
3. slm_block_store<T,N>(offset, val)       — 写入共享内存
4. barrier()                               — 工作组同步
5. slm_block_load<T,N>(offset)             — 从共享内存读取
6. simd<T,N> 向量运算                       — ESIMD SIMD 计算
7. esimd::detail::sum<T,T,N>(vec)          — Subgroup 规约
8. block_store<T,N>(ptr, val)              — 写回全局内存
9. handler.parallel_for(nd_range<2>, SYCL_ESIMD_KERNEL)
```

## 用途

作为自研 llama.cpp 融合 kernel（dequant+GEMM、QKV fusion 等）的代码模板。
