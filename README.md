# SYCL Fusion Project

Intel Arc A770 XMX 融合 kernel 加速 llama.cpp — 集成 IPEX-LLM 的 GPU 端 dequant+GEMM 融合技术。

## 目标

将 llama.cpp SYCL 后端的 Qwen2.5-7B Q4_K_M 推理从 **26.4 t/s → 40 t/s** (+52%)，
通过 GPU 端反量化+GEMM 融合 kernel 消除 CPU 端瓶颈。

## 项目结构

```
sycl-fusion-project/
├── llama.cpp/              ← llama.cpp 源码 (fusion-dev 分支, 含 DPAS)
├── ipex-kernels/           ← IPEX-LLM 提取的 48 个 SPIR-V kernel 模块
├── ipex-binary/            ← IPEX-LLM .so 运行时依赖 (dlopen 用)
├── scripts/
│   ├── build.sh            ← 编译脚本 (AOT/JIT)
│   └── benchmark.sh        ← 对比测试脚本
├── docs/
│   └── kernel-signatures.md ← 反编译的 kernel 函数签名
└── README.md
```

## 快速开始

```bash
# 编译 (JIT 模式, 1-2 min)
bash scripts/build.sh

# 编译 (AOT 模式, 15-20 min, 性能最优)
bash scripts/build.sh aot

# 运行快速验证
./build/bin/llama-cli -m /home/w/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    -p "Hello" -n 16 -ngl 99

# Benchmark
./build/bin/llama-bench -m /home/w/models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -ngl 99 -b 2048 -t 4 -n 128
```

## 环境要求

- Intel Arc A770 (acm-g10)
- oneAPI 2025.3 (icpx compiler)
- GPU Driver 24.39
- IPEX-LLM 2.3.0b20250724 二进制 (ipex-binary/)

## 技术路线

| 阶段 | 方法 | 目标 |
|------|------|------|
| 短期 | dlopen IPEX-LLM .so, 调用其 batch_forward_q4_K | 30+ t/s |
| 中期 | 集成 QKV/MLP 融合 kernel (SPIR-V via Level Zero) | 35+ t/s |
| 长期 | 纯 SYCL ESIMD 自研全部 fusion kernel | 40+ t/s, 独立 |

## 模型兼容性

| 模型 | 我们的 build | IPEX-LLM |
|------|:---:|:---:|
| Qwen2.5-7B (qwen2) | ✅ | ✅ |
| Qwen3-14B (qwen3) | ✅ | ✅ |
| Qwen3.5-9B (qwen35) | ✅ | ❌ |
| Qwen3.6-35B-A3B MoE (qwen35moe) | ✅ | ❌ |
| DeepSeek-R1-Distill-Qwen-14B | ✅ | ✅ |
