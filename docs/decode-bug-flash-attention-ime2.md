# Why does llama.cpp decode produce duplicated tokens on SpacemiT K3 (IME2 / A100)?

**Short answer.** The flash-attention kernel that the SpacemiT IME2 backend uses for token-by-token
decode leaves the running maximum uninitialised for all but the first head of a group:

```cpp
// ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:669
// flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<QLEN>
float M[QLEN] = { -INFINITY };  // maximum KQ value   <- only M[0] is -INF, M[1..QLEN-1] are 0
```

A one-line fix ([patch](../patches/0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch))
makes token-by-token decode as accurate as the batched path (KLD to F16 **0.0667** instead of **0.286**,
over 20 chunks) and removes the duplicated tokens, at **5.12 tok/s** for Qwen2.5-7B Q4_K_M. Without
patching, `-fa off` avoids the bug at −11 % prompt speed.

Labels: **[V]** measured on the board, **[I]** inference or calculation, **[NV]** not verified by us.
All runs: K3 Pico-ITX 8 GB, Bianbu 4.0.1, kernel 6.18.3-1.0.8.4, X100 2.0 GHz / A100 1.8 GHz,
`cpufreq/boost = 0`, October 26–27, 2026, unless stated otherwise.

- [Who is affected](#who-is-affected)
- [Symptoms in generated text](#symptoms-in-generated-text)
- [How large the error is](#how-large-the-error-is)
- [Root cause](#root-cause)
- [The fix and what it changes](#the-fix-and-what-it-changes)
- [Workaround without patching: `-fa off`](#workaround-without-patching--fa-off)
- [Related: `thread_n exceeds perfer_core_ids size 8`](#related-thread_n-exceeds-perfer_core_ids-size-8)
- [Status upstream](#status-upstream)
- [Reproduce](#reproduce)

## Who is affected

Anyone who generates text on the A100 cores of a SpacemiT K3 through the IME2 path of llama.cpp:

| build | llama.cpp inside | how we obtained it | decode defect |
|---|---|---|---|
| `llama.cpp-tools-spacemit` 0.0.8 | build 8373 (a5b514468) | Launchpad PPA `spacemit/k3`, unpacked to a separate dir, `LD_LIBRARY_PATH` | yes [V] |
| `llama.cpp-tools-spacemit` 0.1.1 | 17ce6aa | Launchpad PPA `spacemit/k3`, same way | yes [V] |
| `llama.cpp-tools-spacemit` 0.1.9 | build 10050 (5496cdc9c) | Bianbu `resolute-porting`, installed (`/usr/bin/llama-*`) | yes [V] |
| upstream llama.cpp `0a1ef30` (b11796) built with `-DGGML_CPU_RISCV64_SPACEMIT=ON` | — | built on the board | yes [V], same result as 0.1.9 to the 6th digit |
| upstream master `f5222f2` (2026-10-28) | — | source only | the same line is present [V], not run |

0.1.1 and 0.1.9 give results identical to the last digit; 0.0.8 is within 0.6 %. Thread count does not
matter (0.1.9 default vs `-t 8` — identical), runs are deterministic, the defect is present on
Qwen2.5 1.5B, 3B and 7B. Versions 0.1.7, 0.1.6, 0.1.3 and 0.0.9 could not be tested: the Bianbu archive
was unreachable on 2026-10-26 (see [clocks and firmware](clocks-boost-firmware-reliability.md#package-archive)).

Not affected: **embeddings** (single pass, no decode; see [embeddings](embeddings-server.md)), and the
generic CPU path of llama.cpp on the X100 cores. Other K3 boards (Milk-V Jupiter 2, K3-CoM260) use the
same SoC and the same backend, so they should behave the same [I]; we tested only the K3 Pico-ITX.

## Symptoms in generated text

`llama-server` `/completion`, temperature 0, `top_k` 1, the same prompt tokens on both builds (token
counts equal for all 8 prompts), Qwen2.5-7B-Instruct Q4_K_M [V]:

- each build reproduces itself 8 of 8; vendor 0.1.9 vs upstream CPU — **0 of 8** answers identical;
- one-word classification: upstream `network_config`; vendor `network\nnetwork\nnetwork\nuser`
  (reference probability of `_config` 1.000, vendor 0.056);
- translation to English: `cloudss`, `APIss`, `the-the technologicalological generation`;
- SQL index advice: `the the` in almost every sentence;
- extraction to JSON: values wrapped in spaces (`" k3 "`), misspelled keys;
- the same in Russian answers: `хостта`, `хосттом`, `поместитсясь`.

17 such duplication artifacts in the 8 vendor answers, 0 in upstream [V].

## How large the error is

KL divergence from `llama-perplexity --kl-divergence` on ~99 KB of Russian technical prose, `-c 512`
([methodology](methodology.md)). "Token-by-token" is `-b 512 -ub 1` (what text generation does),
"batched" is the default `-ub 512` (what prompt processing does). Full table:
[`data/kld_results.csv`](../data/kld_results.csv).

Qwen2.5-7B-Instruct Q4_K_M against the official F16 GGUF, 20 chunks (~5,100 scored tokens) [V]:

| build | mode | mean KLD | 99 % KLD | max KLD | same top-1 | PPL(Q)/PPL(F16) |
|---|---|---|---|---|---|---|
| upstream generic CPU (X100) | token-by-token | 0.0463 | 0.257 | 1.08 | 87.86 % | 0.993 |
| upstream generic CPU (X100) | batched | 0.0465 | 0.267 | 1.17 | 87.92 % | 0.992 |
| **vendor 0.1.9 (A100, IME2)** | **token-by-token** | **0.2859** | 2.15 | 13.7 | **76.75 %** | **1.144** |
| vendor 0.1.9 = upstream + IME2 | batched | 0.0663 | 0.403 | 1.81 | 85.63 % | 0.999 |
| upstream + IME2 **+ fix** | token-by-token | **0.0667** | 0.399 | 1.86 | **85.77 %** | 0.998 |

Token-by-token, the vendor path is 6.2 times further from F16 than the generic CPU build. Perplexity is a poor judge here: on this text
Q4_K_M has *lower* perplexity than F16 (13.58 vs 13.69), so decisions are made on KLD.

Smaller models (reference: upstream CPU, same model, token-by-token, 4 chunks) [V]:

| model | vendor batched: KLD / top-1 | vendor token-by-token: KLD / top-1 | with fix, token-by-token |
|---|---|---|---|
| Qwen2.5-3B Q4_K_M | 0.102 / 87.3 % (max 11.5) | 0.110 / 86.8 % | 0.103 / 86.7 % (max 11.2) |
| Qwen2.5-1.5B Q4_K_M | 0.063 / 86.4 % | **0.245 / 74.4 %** | **0.062 / 86.2 %** |

The 3B model stays at ~0.10 in both modes with the fix: that is the Q4_K repack loss, not attention
([quantization page](q4k-repack-and-choosing-a-quant.md)).

## Root cause

The IME2 backend has its own flash attention. Batched evaluation goes through
`forward_flash_attn_ext_f16_tiled_vlen1024_vf16`; token-by-token evaluation goes through
`…_one_chunk_vlen1024_vf16` → `flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4|2>` / `_m1`.
The `_mrow<QLEN>` variant processes QLEN consecutive query heads that share one KV head (GQA):

```cpp
float S[QLEN] = { 0.0f };       // sum
float M[QLEN] = { -INFINITY };  // maximum KQ value
```

Aggregate initialisation sets only `M[0]` to −∞; `M[1..QLEN-1]` are zero. For heads 1..QLEN-1 the online
softmax therefore starts with a running maximum of 0. While their attention scores are negative, the
weights `exp(s − 0)` are small; they are converted to `_Float16` for the fp16 V accumulator and underflow
to subnormals or zero, while the sum `S` keeps accumulating in float. The head output is scaled down and
distorted. The tiled kernel sets `M[i] = -INFINITY` in a loop, which is why batched evaluation is fine.

Which heads are hit depends on grouping: the decode loop takes 4 (or 2) consecutive query heads into one
`_mrow<4>` (`_mrow<2>`) call when they share a KV head and fall into the same thread's range, and the rest
go through `_m1`. Every head except the first of each `_mrow` call starts with `M = 0`. Qwen2.5-7B has 28
query heads over 4 KV heads (7 per KV head), so a large share of heads is affected on every decoded token
[I, from the code].

How it was localised [V]:

1. An upstream build with the IME2 backend reproduces the vendor package to the 6th digit
   (token-by-token vs upstream CPU: 0.214828 / 81.961 % for both; batched vs F16: 0.066291 / 85.627 %),
   so the defect is in the open-source backend, not in a closed part of the package.
2. A kernel stand (matmul on real weights, [reproduce/ime2-kernel-check](../reproduce/ime2-kernel-check/))
   shows that the one-row and four-row IME2 matmul kernels give identical results for Q4_K and Q6_K
   (and agree within quantization noise for Q4_0): the decode-only defect is not in matmul.
3. Switching flash attention off removes it:

| upstream + IME2, Q4_K_M 7B, vs upstream CPU | FA `auto` (= on) | `-fa off` |
|---|---|---|
| token-by-token, 2 chunks | 0.214828 / 81.96 % | **0.028893 / 91.57 %** |
| batched, 4 chunks | 0.033830 / 89.51 % | 0.033940 / 90.49 % |
| control: upstream CPU token-by-token, `-fa off` vs FA on | — | 0.004741 / 97.45 % |

## The fix and what it changes

```diff
--- a/ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp
+++ b/ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp
@@ -666,7 +666,8 @@
     GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
     float S[QLEN] = { 0.0f };       // sum
-    float M[QLEN] = { -INFINITY };  // maximum KQ value
+    float M[QLEN];                  // maximum KQ value
+    std::fill(M, M + QLEN, -INFINITY);
```

`<algorithm>` is already included by the file. The patch applies unchanged to `0a1ef30` and to master
`f5222f2` (the file is byte-identical in both, checked 2026-10-28) [V].

With the fix, Qwen2.5-7B Q4_K_M [V]:

| measurement | value |
|---|---|
| token-by-token vs upstream CPU, 2 chunks | **0.028971 / 91.765 %** (= `-fa off`) |
| token-by-token vs F16, 20 chunks | **0.0667 / 85.77 %** (batched IME2: 0.0663 / 85.63 %) |
| token-by-token vs F16, first 4 chunks | 0.0658 / 86.18 % (vendor without fix 0.2841 / 77.45 %, upstream CPU 0.0426 / 88.53 % — cumulative values at chunk 4 of the 20-chunk runs) |
| `llama-bench` pp512 / tg128 | **40.09 / 5.12 tok/s** (unpatched: 38.48 / 5.08; `-fa off`: 35.55 / 4.98) |
| the 8 greedy prompts | duplication artifacts **0** (vendor 17); 2 of 8 answers byte-identical with upstream (classification, JSON extraction) |

The other 6 greedy answers diverge from upstream at points where upstream itself is uncertain (top-1/top-2
probabilities 0.42–0.57 vs 0.18–0.46), except one (0.90/0.10 at token 68). That is ordinary quantization
noise between two different matmul paths, not the defect.

The remaining distance to F16 (0.067 vs 0.046 for the generic CPU path) is the Q4_K weight repack, which
the fix does not touch. Using Q4_0 (default `llama-quantize` mix, output tensor Q6_K) instead gives
**0.032** at **5.97 tok/s** — see [which quantization to use](q4k-repack-and-choosing-a-quant.md).

## Workaround without patching: `-fa off`

`llama-cli`/`llama-server`/`llama-perplexity -fa off` gives the same quality as the fix (0.0289 vs
0.0290 above) on the upstream IME2 build [V]. Cost: prompt processing 35.55 vs 40.09 tok/s (−11 %),
decode 4.98 vs 5.12 tok/s (−3 %). The vendor package runs the same code, so `-fa off` should work there
too [I, not measured on the package].

## Related: `thread_n exceeds perfer_core_ids size 8`

Not the decode bug, but on the same path: `llama.cpp-tools-spacemit` 0.0.8 and 0.1.1 abort at the default
thread count (16, the number of CPUs) [V]:

```
ime.cpp:1450: thread_id 9 exceeds perfer_core_ids size 8     (0.0.8)
ime.cpp:1704: thread_n 10 exceeds perfer_core_ids size 8     (0.1.1)
```

They work with `-t 8`; 0.1.9 handles the default itself. Upstream reports of the same message:
[ggml-org/llama.cpp#29261](https://github.com/ggml-org/llama.cpp/issues/29261) (open) and the proposed
[ggml-org/llama.cpp#30284](https://github.com/ggml-org/llama.cpp/pull/30284) "select A100 cores for
SpaceMiT default threads" (open, 2026-10-28).

## Status upstream

As of 2026-10-28 [V]: the line is unchanged in `ggml-org/llama.cpp` master (`f5222f2`), and a search for
"spacemit" in the issues and pull requests of `ggml-org/llama.cpp` shows no report of this bug. The
finding was posted on the SpacemiT developer forum:
[forum.spacemit.com/t/topic/1733](https://forum.spacemit.com/t/topic/1733) (in Chinese).

## Reproduce

See [reproduce/README.md](../reproduce/README.md#1-decode-defect-kld-token-by-token-vs-batched). In short:

```bash
# reference logits: any generic CPU build, token-by-token
llama-perplexity -m qwen2.5-7b-instruct-q4_K_M.gguf -f text.txt -c 512 --chunks 2 -b 512 -ub 1 -t 8 \
    --kl-divergence-base ref_ub1.kld
# IME2 build (vendor /usr/bin/llama-perplexity or upstream with -DGGML_CPU_RISCV64_SPACEMIT=ON), same mode
llama-perplexity -m qwen2.5-7b-instruct-q4_K_M.gguf -f text.txt -c 512 --chunks 2 -b 512 -ub 1 \
    --kl-divergence-base ref_ub1.kld --kl-divergence
# without the fix: Mean KLD ~0.21, "Same top p" ~82 %; with the fix or -fa off: ~0.03 / ~92 %
```

Any text reproduces it; ours is not published (internal documentation), wikitext-2 should work as well.
