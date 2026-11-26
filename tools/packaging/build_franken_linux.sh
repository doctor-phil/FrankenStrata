#!/usr/bin/env bash
# build_franken_linux.sh — Build and box a self-contained Linux FrankenStrata bundle
# containing the HIP primary engine (TheRock wheels), SYCL agent (oneAPI / Docker),
# HIP + CPU agents, hardware profiler, and unified `franken-strata` launcher.
#
# Environment variables:
#   HIP_ARCH=gfx1100                 AMD HIP target arch (default: gfx1100)
#   SYCL_AOT=bmg-g31                 Intel SYCL AOT target (empty = SPIR-V JIT; default: bmg-g31)
#   SYCL_DOCKER_IMAGE=strata-sycl-dev:latest
#   LLAMA_DIR=<path>                 Optional local llama.cpp checkout for -DSTRATA_GGML_DIR
#   OUT=<dir>                        Output bundle folder (default: $PWD/dist/frankenstrata-linux-x86_64)
#   TARBALL=<path>                   Optional .tar.gz output path
#   SKIP_BUILD=1                     Reuse existing build-hip-franken and build-sycl-franken
#   JOBS=<N>                         Parallel build jobs (default: nproc/2)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HIP_ARCH="${HIP_ARCH:-gfx1100}"
SYCL_AOT="${SYCL_AOT:-bmg-g31}"
SYCL_DOCKER_IMAGE="${SYCL_DOCKER_IMAGE:-strata-sycl-dev:latest}"
OUT="${OUT:-$REPO/dist/frankenstrata-linux-x86_64}"
JOBS="${JOBS:-$(( $(nproc) / 2 ))}"
[ "$JOBS" -lt 1 ] && JOBS=1
HIP_BUILD="${HIP_BUILD:-$REPO/build-hip-franken}"
SYCL_BUILD="${SYCL_BUILD:-$REPO/build-sycl-franken}"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  ROOT="$("$REPO/.venv/bin/rocm-sdk" path --root)"
  SP="$(dirname "$ROOT")"
  LIB_PKG="$(ls -d "$SP"/_rocm_sdk_libraries_* | head -n 1)"
  BITCODE="$ROOT/lib/llvm/amdgcn/bitcode"
  [ -d "$BITCODE" ] || BITCODE="$ROOT/amdgcn/bitcode"
  export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH="$ROOT" HIP_PATH="$ROOT"
  export LD_LIBRARY_PATH="$ROOT/lib:$LIB_PKG/lib:${LD_LIBRARY_PATH:-}"
  export PATH="$REPO/.venv/bin:$ROOT/bin:$ROOT/llvm/bin:$PATH"

  GGML_ARG=()
  [ -n "${LLAMA_DIR:-}" ] && GGML_ARG=("-DSTRATA_GGML_DIR=$LLAMA_DIR")

  cmake -G Ninja -S "$REPO/engine" -B "$HIP_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
    -DCMAKE_HIP_COMPILER="$ROOT/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROOT" \
    -DCMAKE_PREFIX_PATH="$ROOT;$LIB_PKG" \
    -DCMAKE_HIP_FLAGS="--rocm-path=$ROOT --rocm-device-lib-path=$BITCODE" \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" \
    "${GGML_ARG[@]}"
  cmake --build "$HIP_BUILD" -j "$JOBS" --target strata franken-agent-hip franken-agent-cpu franken-profile

  DOCKER_GGML_MOUNT=()
  DOCKER_GGML_ARG=""
  if [ -n "${LLAMA_DIR:-}" ]; then
    DOCKER_GGML_MOUNT=(-v "$LLAMA_DIR:/ggml:ro")
    DOCKER_GGML_ARG="-DSTRATA_GGML_DIR=/ggml"
  fi
  docker run --rm --device /dev/dri \
    -v "$REPO:/src" \
    "${DOCKER_GGML_MOUNT[@]}" \
    "$SYCL_DOCKER_IMAGE" \
    "set -euo pipefail; \
     cmake -S /src/engine/sycl -B /src/$(basename "$SYCL_BUILD") -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
       ${SYCL_AOT:+-DSTRATA_SYCL_AOT=$SYCL_AOT} -DSTRATA_SYCL_PARITY=OFF $DOCKER_GGML_ARG; \
     cmake --build /src/$(basename "$SYCL_BUILD") -j $JOBS --target franken-agent-sycl"
fi

TAR_ARGS=()
[ -n "${TARBALL:-}" ] && TAR_ARGS=(--tarball "$TARBALL")

"$PY" "$REPO/tools/packaging/package_franken.py" \
  --hip-build "$HIP_BUILD" \
  --sycl-build "$SYCL_BUILD" \
  --sycl-docker-image "$SYCL_DOCKER_IMAGE" \
  --arch "$HIP_ARCH" \
  --sycl-aot "$SYCL_AOT" \
  --out "$OUT" \
  "${TAR_ARGS[@]}"
