# IPEX-LLM 融合 Kernel 集成指南

## 概览

三层架构，从简单到复杂：

```
Layer 1 (dlopen):        IPEX .so 函数调用      → 2 处修改, 立即可用
Layer 2 (SPIR-V):        Level Zero kernel 加载  → 1 个新文件, 需编译链接
Layer 3 (Graph):         GGML 图重写             → 架构级改动, 长期目标
```

## Layer 1 集成: ggml-sycl.cpp 修改点

### 修改点 1: 头部添加 include + init 调用

**位置**: 文件顶部, 在 `static bool g_sycl_loaded = false;` 之前

```cpp
// === 新增 ===
#include "ggml-sycl-ipex.h"

// === 在 ggml_sycl_init() 或首次 backend init 位置添加 ===
// (搜索 "g_ggml_sycl_disable_graph = get_sycl_env" 附近)
ipex_fusion::get_fusion_ctx().init();
```

### 修改点 2: ggml_sycl_op_mul_mat_sycl 中插入快速路径

**位置**: 在 DPAS `#endif` 之后, FP16 `if ((src0->type == GGML_TYPE_F16 ...` 之前

当前代码 (line ~2491):
```cpp
#endif  // GGML_SYCL_USE_DPAS

    // === 在这一行之后插入 ===
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 ...
```

插入后:
```cpp
#endif  // GGML_SYCL_USE_DPAS

    // === IPEX Fusion fast path ===
    if (ipex_fusion::try_fused_mul_mat(
            src0, src1, src1_ddf_i, dst_dd_i,
            row_diff, src1_ncols, ne10, *stream))
    {
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
    // === IPEX Fusion end ===

    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 ...
```

### 修改点 3: CMakeLists.txt 添加新源文件

**位置**: ggml/src/ggml-sycl/CMakeLists.txt, 在 `ggml_add_backend_library` 之后

```cmake
# IPEX Fusion integration sources
target_sources(ggml-sycl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-sycl-ipex-dlopen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-sycl-ipex-spirv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-sycl-ipex-graph.cpp
)

# dlopen needs libdl, SPIRV loader needs Level Zero
target_link_libraries(ggml-sycl PRIVATE dl)
# Level Zero is already linked via GGML_SYCL_SUPPORT_LEVEL_ZERO
```

## Layer 2 集成: SPIR-V 内核初始化

Layer 2 在第一次使用时延迟加载。在 `ggml_sycl_init()` 中添加:

```cpp
// 在 SYCL 设备初始化完成后
if (info.device_count > 0 && g_ggml_sycl_enable_level_zero) {
    sycl::queue q(sycl::gpu_selector_v);
    ipex_fusion::get_fusion_ctx().spirv.load_recommended(q);
}
```

## Layer 3 集成: 图重写

在 `ggml_backend_sycl_graph_compute()` 中, 在 graph 执行之前:

```cpp
// 在 graph compute 入口处
auto fusion_plan = ipex_fusion::analyze_graph(cgraph);
if (fusion_plan.potential_savings() > 0) {
    ipex_fusion::apply_fusion_plan(cgraph, fusion_plan);
}
```

图重写需要修改 llama.cpp 的 attention/MLP 实现以支持融合输出格式,
这是最复杂的部分, 建议在 Layer 1+2 验证性能收益后再着手。

## 编译

```bash
cd /home/w/sycl-fusion-project
bash scripts/build.sh aot
```

预期新增编译单元: 3 个 `.cpp` 文件
预期链接新增: `-ldl`
编译时间增加: ~30 秒 (JIT) / ~2 分钟 (AOT)

## 运行时

```bash
# Layer 1 需要 IPEX-LLM .so
export IPEX_SO_PATH=/home/w/sycl-fusion-project/ipex-binary/libggml-sycl.so
# Layer 2 需要 SPIR-V kernel 文件
export IPEX_KERNEL_DIR=/home/w/sycl-fusion-project/ipex-kernels

./build/bin/llama-bench -m /home/w/models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -ngl 99 -b 2048 -t 4 -n 128
```

启动日志中应能看到:
```
[IPEX-dlopen] Loaded: q4_K=0x... q5_K=0x... batch_q4k=0x...
[IPEX-dlopen] Converted weights: 2348810240 → 4697620480 bytes (type=15)
```

## 调试

```bash
# 强制使用 Layer 1 路径 (dlopen)
export IPEX_FUSION_LEVEL=1

# 禁用 IPEX fusion (回退到原始路径)
export IPEX_FUSION_LEVEL=0

# 详细日志
export IPEX_FUSION_DEBUG=1
```
