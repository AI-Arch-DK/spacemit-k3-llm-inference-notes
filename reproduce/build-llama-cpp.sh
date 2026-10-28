#!/bin/bash
# Build upstream llama.cpp natively on a SpacemiT K3 (Bianbu 4.0.x, gcc 15.2) in three variants:
#   <src>/build-cpu        generic CPU backend with RVV (run it on X100: taskset -c 0-7 ... -t 8)
#   <src>/build-ime        + SpacemiT IME2 backend (moves its workers to the A100 cores by itself)
#   <src>-fafix/build-fafix  the IME2 build with patches/0001 (flash-attention decode fix), FIX=1 only
# Usage: [REV=0a1ef30] [SRC=llama.cpp] [FIX=1] [JOBS=8] bash build-llama-cpp.sh
# The IME2 build took ~12 min on the 8 X100 cores.
set -euo pipefail
REV=${REV:-0a1ef30}
SRC=${SRC:-llama.cpp}
FIX=${FIX:-1}
JOBS=${JOBS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/../patches/0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch
TARGETS=(--target llama-perplexity llama-bench llama-server llama-quantize llama-cli)
RV=(-DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZICBOP=ON)

[ -d "$SRC/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$SRC"
git -C "$SRC" checkout -q "$REV"

cmake -S "$SRC" -B "$SRC/build-cpu" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON "${RV[@]}"
taskset -c 0-7 cmake --build "$SRC/build-cpu" -j "$JOBS" "${TARGETS[@]}"

IME=(-DCMAKE_BUILD_TYPE=Release -DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_CPU_REPACK=OFF -DLLAMA_OPENSSL=OFF
     "${RV[@]}" -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON)
cmake -S "$SRC" -B "$SRC/build-ime" "${IME[@]}"
grep -E "SPACEMIT_RISCV_COMPILER_SUPPORT|RISCV64_SPACEMIT" "$SRC/build-ime/CMakeCache.txt" | grep -v "^//" || true
taskset -c 0-7 cmake --build "$SRC/build-ime" -j "$JOBS" "${TARGETS[@]}"

if [ "$FIX" = 1 ]; then
    FX=$SRC-fafix
    [ -d "$FX" ] || git -C "$SRC" worktree add --detach "$(realpath -m "$FX")" "$REV"
    if git -C "$FX" apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "patch already applied in $FX"
    else
        git -C "$FX" apply "$PATCH"
    fi
    cmake -S "$FX" -B "$FX/build-fafix" "${IME[@]}"
    taskset -c 0-7 cmake --build "$FX/build-fafix" -j "$JOBS" "${TARGETS[@]}"
fi
echo BUILD_DONE
