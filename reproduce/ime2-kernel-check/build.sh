#!/bin/bash
# Build the ime2_check stand on the board with the same gcc and flags as ggml-cpu in the IME2 build
# (<build>/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make). refs_wrap.cpp is compiled with hidden visibility:
# its copies of the kernels must not interpose the libggml-cpu.so kernels that the stand checks.
# Usage: LLAMA_DIR=<llama.cpp checkout> BUILD_DIR=<llama.cpp>/build-ime bash build.sh
set -e
L=${LLAMA_DIR:?set LLAMA_DIR to the llama.cpp source tree}
B=${BUILD_DIR:-$L/build-ime}/bin
S=$L/ggml/src/ggml-cpu/spacemit
cd "$(dirname "$0")"
F="-O3 -DNDEBUG -std=gnu++17 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii -mabi=lp64d
   -DGGML_USE_CPU_RISCV64_SPACEMIT -DRISCV64_SPACEMIT_IME1 -DRISCV64_SPACEMIT_IME2 -D_GNU_SOURCE -D_XOPEN_SOURCE=600
   -I$S -I$L/ggml/src -I$L/ggml/src/ggml-cpu -I$L/ggml/include"
g++ $F -fvisibility=hidden -w -c refs_wrap.cpp -o refs_wrap.o
g++ $F -c ime2_check.cpp -o ime2_check.o
g++ ime2_check.o refs_wrap.o -o ime2_check -L$B -lggml-base -lggml-cpu -ldl -fopenmp -Wl,-rpath,$B
echo BUILD_OK
