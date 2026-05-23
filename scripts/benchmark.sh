#!/bin/bash
# Benchmark harness for SYCL fusion project
# Compares our build vs IPEX-LLM on models both support
# Usage: bash scripts/benchmark.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
IPEX_DIR="$PROJECT_ROOT/ipex-binary"
MODELS=/home/w/models

source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

echo "============================================"
echo "SYCL Fusion Project — Benchmark"
echo "============================================"

# Our build
echo ""
echo "=== Our SYCL Build ==="
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/intel/oneapi/dnnl/2025.3/lib:/opt/intel/oneapi/umf/1.0/lib:/opt/intel/oneapi/mkl/2025.3/lib
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ZES_ENABLE_SYSMAN=1
export GGML_SYCL_PRIORITIZE_DMMV=0

for MODEL in qwen2.5-7b-instruct-q4_k_m.gguf Qwen2.5-0.5B-Instruct-Q4_K_M.gguf; do
    if [ -f "$MODELS/$MODEL" ]; then
        echo "  Model: $MODEL"
        "$BUILD_DIR/bin/llama-bench" -m "$MODELS/$MODEL" -ngl 99 -b 2048 -t 4 -n 128 2>&1 | grep "tg128\|pp512" || echo "  FAILED"
    fi
done

# IPEX-LLM (if stable)
echo ""
echo "=== IPEX-LLM (reference) ==="
export LD_LIBRARY_PATH=$IPEX_DIR
export SYCL_CACHE_PERSISTENT=1
unset GGML_SYCL_PRIORITIZE_DMMV

for MODEL in qwen2.5-7b-instruct-q4_k_m.gguf Qwen2.5-0.5B-Instruct-Q4_K_M.gguf; do
    if [ -f "$MODELS/$MODEL" ]; then
        echo "  Model: $MODEL"
        timeout 60 "$IPEX_DIR/../llama-bench-bin" -m "$MODELS/$MODEL" -ngl 99 -b 2048 -t 4 -n 128 2>&1 | grep "tg128\|pp512" || echo "  FAILED (may segfault)"
    fi
done 2>/dev/null || true

echo ""
echo "Done."
