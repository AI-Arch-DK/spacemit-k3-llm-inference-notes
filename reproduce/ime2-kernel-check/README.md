# ime2-kernel-check: IME2 matmul accuracy on real weights

A small stand that runs the SpacemiT IME2 matmul kernels from a **built** `libggml-cpu.so` (loaded with
`dlsym`, so it checks the exact machine code `llama-perplexity` runs) on the first 256 rows of real
tensors of `qwen2.5-7b-instruct-q4_K_M.gguf`, after the backend's own weight repack, and compares them with:

- the exact result in double (weights dequantized by stock ggml);
- the upstream generic CPU kernels (`ggml_vec_dot_q*_K_q8_K_generic`, `…q4_0_q8_0_generic`);
- the vendor's reference kernels `gemm_kernel_*_mrow_ref` (disabled with `#if 0` in the dispatcher;
  `refs_wrap.cpp` instantiates them from the same source tree).

Written for llama.cpp `0a1ef30` built with `-DGGML_CPU_RISCV64_SPACEMIT=ON` (see `../build-llama-cpp.sh`).
Results from our board: [`data/ime2_matmul_error.tsv`](../../data/ime2_matmul_error.tsv), discussed in
[docs/q4k-repack-and-choosing-a-quant.md](../../docs/q4k-repack-and-choosing-a-quant.md#kernel-stand-matmul-error-on-real-weights).

```bash
# 1. dump weight rows (needs llama.cpp/gguf-py and numpy)
GGUF_PY=llama.cpp/gguf-py MODEL=qwen2.5-7b-instruct-q4_K_M.gguf python3 dump_tensors.py tensors
# 2. build with the same flags as ggml-cpu in the IME2 build
LLAMA_DIR=$PWD/llama.cpp BUILD_DIR=$PWD/llama.cpp/build-ime bash build.sh
# 3. run (it makes itself an A100 thread via /proc/set_ai_thread)
./ime2_check tensors llama.cpp/build-ime/bin/libggml-cpu.so > ime2_check.tsv
```

Columns: relative L2 error against the exact result of `up` (generic CPU), `aq` (activation quantisation
only), `rp` (Q4_K repack formula only), `ex` (lossless kernel on repacked weights), `ref1`/`ref4` (vendor
reference kernels), `m1`/`m4` (built kernels, one row / four rows); `m1_vs_ref1` etc. compare kernels with
each other; `zp_clamped` counts sub-blocks whose zero point hit the 0..15 clamp. Activation kinds:
`norm` (RMSNorm output with real norm weights), `swiglu`, `gauss`, `pos` (all of one sign, worst case).

The comments were translated to English and the board paths replaced by arguments; the code is otherwise
the one that produced our numbers.
