# Which quantization should I use on the SpacemiT K3 IME2 path? (Q4_K_M vs Q4_0, `make_block_q4_1x32`)

**Short answer.** On the A100 / IME2 path, use **Q4_0 with the default `llama-quantize` mix** (output
tensor in Q6_K), not Q4_K_M. For Qwen2.5-7B-Instruct with the [flash-attention fix](decode-bug-flash-attention-ime2.md):
KLD to F16 **0.032** at **5.97 tok/s** (Q4_0) versus 0.066 at 5.12 tok/s (Q4_K_M). The IME2 backend
repacks Q4_K weights into a format with an *integer* zero point per 32-weight sub-block, which costs
6–13.5 % relative error in every matmul; Q4_0 and Q6_K are repacked without loss (Q8_0 uses the same
`i8i8` kernel as Q6_K, so it should be too [I]).

Labels: **[V]** measured, **[I]** inference, **[NV]** not verified. Board: K3 Pico-ITX 8 GB, X100 2.0 GHz /
A100 1.8 GHz, boost off, llama.cpp `0a1ef30` + `GGML_CPU_RISCV64_SPACEMIT=ON` + the FA fix, 2026-10-26/27.

- [Comparison of formats](#comparison-of-formats)
- [Why Q4_K loses precision: `make_block_q4_1x32`](#why-q4_k-loses-precision-make_block_q4_1x32)
- [Kernel stand: matmul error on real weights](#kernel-stand-matmul-error-on-real-weights)
- [What about Q8_0, 3B and embedding models](#what-about-q8_0-3b-and-embedding-models)
- [How to make the Q4_0 file](#how-to-make-the-q4_0-file)

## Comparison of formats

Qwen2.5-7B-Instruct against the official F16 GGUF ([`data/kld_results.csv`](../data/kld_results.csv),
[`data/speed_llama_bench.csv`](../data/speed_llama_bench.csv)) [V]:

| file | path | chunks | KLD token-by-token / top-1 | KLD batched / top-1 | pp512 / tg128, tok/s | file size |
|---|---|---|---|---|---|---|
| Q4_K_M | upstream generic CPU, X100 | 20 | 0.0463 / 87.86 % | 0.0465 / 87.92 % | 9.07 / 3.10 | 4.68 GB |
| Q4_K_M | IME2 + fix, A100 | 20 | 0.0667 / 85.77 % | 0.0663 / 85.63 % ¹ | 40.09 / 5.12 | 4.68 GB |
| **Q4_0, default mix (output Q6_K)** | **IME2 + fix, A100** | 4 | **0.0317 / 91.57 %** | 0.0323 / 91.18 % | **57.54 / 5.97** | 4.43 GB |
| Q4_0, default mix (output Q6_K) | upstream generic CPU, X100 | 4 | 0.0334 / 90.88 % | 0.0340 / 90.20 % | — | 4.43 GB |
| Q4_0 `--pure` (every tensor Q4_0) | IME2 + fix, A100 | 4 | 0.0901 / 84.71 % | 0.0902 / 84.71 % | 58.96 / 6.36 | 4.29 GB |
| Q4_0 `--pure` | upstream generic CPU, X100 | 4 | 0.0896 / 84.90 % | 0.0900 / 84.12 % | — | 4.29 GB |

¹ Batched value of the unpatched IME2 build (identical to the vendor package); the fix does not touch the
batched kernel [I]. For the 4-chunk rows, the matching Q4_K_M numbers are 0.0658 (IME2 + fix) and 0.0426
(upstream CPU, cumulative value at chunk 4 of the 20-chunk run).

What this says:

- On Q4_0 the IME2 path adds no error beyond the quantization itself (0.0317 vs 0.0334 on the CPU path).
- Q4_0 with the default mix is better than Q4_K_M on *either* path here (0.032 vs 0.043 on CPU), and
  +17 % decode / +44 % prompt against Q4_K_M on IME2. Keeping the output tensor in Q6_K costs 6 % of
  decode speed against `--pure` and cuts KLD almost threefold.
- Caveat [I]: the Q4_0 files were quantized from the same F16 that serves as the reference, while the
  Q4_K_M file has a different origin (a published GGUF, sha256 `2bada8a7…`). Part of the Q4_0 advantage
  may come from that, not from the type.

## Why Q4_K loses precision: `make_block_q4_1x32`

`ggml/src/ggml-cpu/spacemit/repack.cpp`, `make_block_q4_1x32` converts each Q4_K sub-block minimum `m`
into an **integer** zero point `round(m / d)`, clamped to 0..15, because the IME2 kernel works with
`int8 × int4` and adds the zero point as `asum · zp` in int16. The rounding shift (up to half a
quantization step) is applied to all 32 weights of the sub-block at once and does not average out over
the dot product.

- The clamp never fired: 0 of 28,672 sub-blocks per tensor [V].
- Computing the result with this formula in double matches the kernels to 0.05–0.07 % [V]: the kernels
  compute exactly what they are given; the loss is in the format.
- It affects batched and token-by-token evaluation equally; it is the remaining gap after the FA fix
  (0.066 vs 0.046 to F16).
- It cannot be fixed by changing the formula alone: the kernel only supports an integer zero point.
  Options are Q4_0 (above) or a kernel change that keeps the sub-block minimum, for example as fp16 [I].
  The question is open on the [SpacemiT forum, topic 1733](https://forum.spacemit.com/t/topic/1733).

## Kernel stand: matmul error on real weights

[`reproduce/ime2-kernel-check`](../reproduce/ime2-kernel-check/) runs the IME2 kernels from the built
`libggml-cpu.so` (via `dlsym`, i.e. the exact code `llama-perplexity` runs) on the first 256 rows of real
tensors of layers 0/14/27 of `qwen2.5-7b-instruct-q4_K_M.gguf`, after the backend's own repack, and
compares them with the exact result (double, weights dequantized by stock ggml) and with the vendor's
reference kernels `*_mrow_ref` (disabled with `#if 0` in the dispatcher, instantiated from the same
source). Activations are synthetic: RMSNorm output with the layer's real norm weights, SwiGLU, Gaussian,
and "all of one sign" as the worst case. Error = relative L2 over 4×256 outputs
([`data/ime2_matmul_error.tsv`](../data/ime2_matmul_error.tsv)) [V]:

| weight type → IME2 kernel | upstream generic CPU | IME2 m1 (one token) | IME2 m4 (batch) | m1 vs vendor reference kernel |
|---|---|---|---|---|
| Q6_K → `i8i8` | 0.7–1.6 % | 0.7–1.1 % | = m1 | 0 |
| **Q4_K → `i8i4`** | 0.5–3.2 % | **6–13.5 %** (same-sign activations 20–45 %) | = m1 | 0.1–0.3 % |
| Q4_0 → `i8i4_hp` (fp16 accumulation) | 0.3–1.8 % | 0.4–1.9 % | 0.55–2.6 % | 0.2–1.05 % |

Two side results:

- For Q4_K and Q6_K the one-row and four-row kernels give identical results, and for Q4_0 they agree
  within quantization noise, so the decode-only defect is not in matmul (it is in attention).
- `gemm_kernel_i8i8_m4` has a contract of at most 32 columns per call (`ime.cpp` respects it). Our first
  stand run passed 256 columns and got 93 % error — a bug in the stand, fixed before the numbers above.

## What about Q8_0, 3B and embedding models

- **Q8_0 7B does not fit** in 8 GB on this path [V: 8.1 GB file]: the backend keeps the repacked weights
  in its own memory. The `i8i8` kernel it would use was exact on Q6_K weights on the stand; Q8_0 itself was
  not run [I].
- **Qwen2.5-3B Q4_K_M** with the fix: KLD 0.103 token-by-token and 0.102 batched against upstream on the
  same model, with outliers up to KLD 11 — the Q4_K repack again [V]. 1.5B Q4_K_M is fine (0.062).
  A Q4_0 3B was not measured.
- **Embedding models in Q4_K_M** lose precision visibly: Qwen3-Embedding-4B Q4_K_M on the vendor path
  vs upstream: cosine min **0.958**, median 0.980, top-10 retrieval overlap 86.8 % — against 0.9989 /
  0.9997 / 97.9 % for Qwen3-Embedding-0.6B in Q8_0 [V]. Use Q8_0 (or Q4_0) embedding models on IME2
  ([embeddings page](embeddings-server.md)).

## How to make the Q4_0 file

From the official F16 (`Qwen/Qwen2.5-7B-Instruct-GGUF`, revision `bb5d59e0`, 4 split files; sha256 in
[`data/f16_reference_sha256.txt`](../data/f16_reference_sha256.txt)); `llama-quantize` takes the first
split and finds the rest. On the board it took 58 s:

```bash
llama-quantize qwen2.5-7b-instruct-fp16-00001-of-00004.gguf qwen2.5-7b-instruct-q4_0.gguf Q4_0 8
# --pure would make every tensor Q4_0 (worse: KLD 0.090 instead of 0.032)
```

The F16 file is 15.2 GB; on the 8 GB board it is used through mmap without swap (peak RSS 7.6 GB).
