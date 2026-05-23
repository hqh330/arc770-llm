#!/bin/bash
# Build llama.cpp with DPAS + IPEX-LLM fusion kernel integration
# Usage: bash scripts/build.sh [aot|jit]
#   aot  — AOT compile for Arc A770 (slow build, fast first run)
#   jit  — JIT compile (fast build, ~5s GPU compile on first run) [default]

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

source /opt/intel/oneapi/setvars.sh --force
export PATH=/opt/intel/oneapi/compiler/2025.3/bin:$PATH

AOT_FLAG=""
if [ "$1" = "aot" ]; then
    AOT_FLAG="-DGGML_SYCL_DEVICE_ARCH=acm-g10"
    echo "=== AOT mode (acm-g10) ==="
else
    echo "=== JIT mode ==="
fi

cmake -S "$PROJECT_ROOT/llama.cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_USE_DPAS=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  $AOT_FLAG

echo ""
echo "=== Building ($(nproc) jobs) ==="
cmake --build "$BUILD_DIR" --config Release -j $(nproc)

echo ""
echo "=== Build complete ==="
echo "Binaries: $BUILD_DIR/bin/"
ls "$BUILD_DIR/bin/llama-cli" "$BUILD_DIR/bin/llama-bench" "$BUILD_DIR/bin/libggml-sycl.so"*
