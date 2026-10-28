# SpacemiT K3 LLM inference notes: llama.cpp on the IME2 / A100 cores (K3 Pico-ITX, Bianbu 4.0, RISC-V)

Measured notes on running LLMs and embedding models on a **SpacemiT K3 Pico-ITX** (8 GB, Bianbu 4.0.1)
with **upstream llama.cpp**, the vendor's **`llama.cpp-tools-spacemit`** (A100 cores, IME2 matrix
extension) and **Ollama**, October 2026. It includes a **one-line fix for wrong token-by-token decode on
the IME2 path** (duplicated tokens such as "the the", "cloudss") and the data behind every number.
Independent testing; not affiliated with SpacemiT.

## Key findings

1. **Text generation on the IME2 / A100 path is broken out of the box.** Token-by-token decode in
   `llama.cpp-tools-spacemit` 0.0.8, 0.1.1, 0.1.9 and in upstream llama.cpp built with
   `-DGGML_CPU_RISCV64_SPACEMIT=ON` gives visibly duplicated sub-word tokens. Measured for 0.1.9 against
   the F16 model: mean KL divergence **0.286** instead of 0.046 (generic CPU), same top-1 token 76.7 % vs
   87.9 %; the other builds show the same deviation against the CPU build (KLD 0.214–0.215).
   → [decode bug](docs/decode-bug-flash-attention-ime2.md)
2. **Root cause: `float M[QLEN] = { -INFINITY };`** in `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:669`
   (`flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow`) initialises only `M[0]`. The
   [one-line patch](patches/0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch) makes decode
   match the batched path (KLD **0.0667** over 20 chunks) at **5.12 tok/s**; without patching, `-fa off`
   works at −11 % prompt speed. Still unfixed in llama.cpp master `f5222f2` (2026-10-28).
3. **Use Q4_0, not Q4_K_M, on IME2.** The backend repacks Q4_K with an integer zero point
   (`make_block_q4_1x32`): 6–13.5 % matmul error. Qwen2.5-7B **Q4_0** (default `llama-quantize` mix,
   output Q6_K) with the fix: KLD **0.032**, **5.97 tok/s** decode, **57.5 tok/s** prompt.
   → [quantization](docs/q4k-repack-and-choosing-a-quant.md)
4. **Embeddings are good, memory is not.** Qwen3-Embedding-0.6B Q8_0 on the vendor path matches
   upstream (cosine ≥ 0.9989), **51 ms** per short text locally, **35 ms** from a LAN client. But the vendor
   `llama-server --embedding` does not reuse memory across requests of different lengths: **7.3 GB** at
   `-c 4096`, **OOM-killed** at `-c 8192`; short queries only are stable for 8 h (+53 MB).
   → [embeddings](docs/embeddings-server.md)
5. **Do not enable `cpufreq/boost` for A100 work.** At the turbo OPPs (A100 2.0 GHz) results were randomly
   corrupted (mean KLD 124, then 0.089 in an identical run); at the nominal 1.8 GHz all **60** consecutive
   runs gave the same result statistics. → [clocks, firmware, reliability](docs/clocks-boost-firmware-reliability.md)
6. **One model at a time on 8 GB.** A 7B generator next to the embedding server hung the board
   (thrashing, no swap). The vendor runtime's main thread stays on X100 and slows down an X100 generator
   running alongside by 77 %. → [platform](docs/platform-hmp-memory-runtimes.md)

## Numbers at a glance

Qwen2.5-7B-Instruct, `llama-bench` pp512 / tg128, X100 2.0 GHz / A100 1.8 GHz, boost off, after the
October 2026 firmware update ([all runs](data/speed_llama_bench.csv), [all KLD runs](data/kld_results.csv)):

| path (cores) | quant | prompt pp512, tok/s | decode tg128, tok/s | KLD to F16, token-by-token | output |
|---|---|---|---|---|---|
| upstream llama.cpp, generic CPU (X100 ×8) | Q4_K_M | 9.07 | 3.10 | 0.046 | correct |
| `llama.cpp-tools-spacemit` 0.1.9 (A100, IME2) | Q4_K_M | 42.4 | 5.06 | **0.286** | **duplicated tokens** |
| upstream + IME2 backend + fix (A100) | Q4_K_M | 40.09 | 5.12 | 0.067 | correct |
| upstream + IME2 backend + fix (A100) | **Q4_0 (output Q6_K)** | **57.54** | **5.97** | **0.032** ¹ | correct, recommended |
| upstream + IME2 backend + fix (A100) | Qwen2.5-1.5B Q4_K_M | 181.3 | 20.44 | — ² | correct |
| upstream llama.cpp, generic RVV on A100 | Q4_K_M | 0.51 | 0.46 | — | unusable (factory clocks) |
| Ollama built from source (X100), `num_thread 8` / default 16 | Q4_K_M | — | 2.65 / 0.16–0.22 ³ | — | correct, slow |

¹ 4 chunks; on the same 4 chunks Q4_K_M with the fix gives 0.066 and the generic CPU path 0.043 (Q4_K_M)
/ 0.033 (Q4_0). ² 0.062 against upstream on the same model. ³ `/api/generate`, 150 tokens, factory clocks.
KLD values are over 20 chunks unless marked.

Before the update (factory clocks 2.2 / 1.8 GHz) the vendor path did 42.9 / 5.65 tok/s and upstream on X100
10.0 / 3.41. Other numbers: X100 memory READ 20.8 GB/s, four A100 cores 31.9, both clusters 36.9 of 51.2
theoretical; DRAM latency 151 ns; UFS 586 / 165 MiB/s; 1GbE 941 Mbit/s both ways; 30 min on all 16 cores
≤ 62 °C without throttling.

## Which page answers my question?

| question | page |
|---|---|
| Why does llama.cpp on K3 print duplicated tokens ("the the", "cloudss", "APIss")? | [decode bug](docs/decode-bug-flash-attention-ime2.md) |
| Is there a patch for `rvv_kernels.cpp` / `flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow`? | [patches/](patches/) |
| `ime.cpp: thread_n exceeds perfer_core_ids size 8` — what is it? | [decode bug → related](docs/decode-bug-flash-attention-ime2.md#related-thread_n-exceeds-perfer_core_ids-size-8) |
| Q4_K_M, Q4_0 or Q8_0 on the A100 / IME2 path? | [quantization](docs/q4k-repack-and-choosing-a-quant.md) |
| Can the K3 serve embeddings for RAG? Why does `llama-server --embedding` grow until OOM? | [embeddings](docs/embeddings-server.md) |
| Is `cpufreq/boost` (turbo) safe? What changed with firmware 1.0.8? | [clocks and firmware](docs/clocks-boost-firmware-reliability.md) |
| How do I run a program on the A100 cores (`/proc/set_ai_thread`, HMP)? | [platform](docs/platform-hmp-memory-runtimes.md#two-kinds-of-cores-how-the-a100-cluster-is-reached-hmp) |
| `open(/dev/tcm_sync_mem) failed, errno=2`, `exclude_main_thread`, `vector has been enabled already`? | [platform](docs/platform-hmp-memory-runtimes.md#the-vendor-ai-runtime-tcm-ime2-devtcm_sync_mem) |
| Why is Ollama so slow on K3 (0.2 tok/s)? Does Vulkan on the PowerVR GPU work? | [platform → runtimes](docs/platform-hmp-memory-runtimes.md#runtimes-on-riscv64-llamacpp-ollama-vulkan-node-docker) |
| How were the numbers measured, and what went wrong? | [methodology](docs/methodology.md) |
| How do I reproduce this on my board? | [reproduce/](reproduce/) |

## Tested setup

- **Board:** SpacemiT K3 Pico-ITX, 8 GB LPDDR5, 128 GB UFS 2.2; 8 × X100 (VLEN 256) + 8 × A100 (VLEN 1024,
  IME2), RISC-V RVA23; one sample, headless.
- **OS:** Bianbu 4.0.1 (Ubuntu 26.04 "resolute" base), kernel 6.18.3; factory firmware (U-Boot / OpenSBI
  1.0.1, EC 0.0.19) until 2026-10-21, then `apt full-upgrade` to kernel 6.18.3-1.0.8.4, U-Boot / OpenSBI /
  ESOS 1.0.8, EC 0.0.32.
- **Software:** gcc 15.2; upstream llama.cpp `0a1ef30` (b11796) in generic-CPU, IME2 and IME2+fix builds;
  `llama.cpp-tools-spacemit` 0.1.9 (+ 0.0.8 and 0.1.1 from the Launchpad PPA); `spacemit-runtime` 0.6.2,
  `spacemit-tcm` 3.0.1; Ollama built from source.
- **Models:** Qwen2.5-Instruct 1.5B / 3B / 7B (Q4_K_M, Q4_0; F16 reference from `Qwen/Qwen2.5-7B-Instruct-GGUF`),
  Qwen3-Embedding 0.6B (Q8_0) and 4B (Q4_K_M).
- **Dates:** 2026-10-21/22 and 2026-10-26/27. Labels in the pages: **[V]** measured on the board,
  **[I]** inference or calculation, **[NV]** not verified.

## Repository layout

```
docs/       one page per topic (see the table above)
patches/    the flash-attention decode fix for llama.cpp (ggml-cpu/spacemit)
reproduce/  build script, KLD check, embedding memory test, A100 wrapper, IME2 kernel stand
data/       summary tables (CSV/TSV) rebuilt from the raw reports
```

## Status (2026-10-28)

The faulty line is unchanged in `ggml-org/llama.cpp` master `f5222f2`; no upstream issue or pull request
for it was found. The finding is posted on the SpacemiT developer forum:
[forum.spacemit.com/t/topic/1733](https://forum.spacemit.com/t/topic/1733) (Chinese). Related upstream
items: [ggml-org/llama.cpp#29261](https://github.com/ggml-org/llama.cpp/issues/29261) (abort with
`thread_n exceeds perfer_core_ids size 8`) and
[ggml-org/llama.cpp#30284](https://github.com/ggml-org/llama.cpp/pull/30284) (A100 cores for default threads).

## License

Text and data: [CC BY 4.0](LICENSE). Scripts, the kernel stand and the patch: [MIT](LICENSE-CODE).
To cite: see [CITATION.cff](CITATION.cff).

## See also (external, not verified by us)

- [spacemit-com/K3-Ubuntu-Images](https://github.com/spacemit-com/K3-Ubuntu-Images) — Ubuntu image for the K3 Pico-ITX
- [ganboing/K3-Docs](https://github.com/ganboing/K3-Docs) — collection of K3 SoC and board documents
- [brucehoult/k3_ai](https://github.com/brucehoult/k3_ai) — utility to start a program on the A100 cores
- [sanderjo/SpacemiT-K3-X100-A100](https://github.com/sanderjo/SpacemiT-K3-X100-A100) — notes on the X100 / A100 cores

## 中文摘要（机器翻译，以英文为准）

在 SpacemiT K3 Pico-ITX（8 GB，Bianbu 4.0.1）上，用上游 llama.cpp、厂商的 `llama.cpp-tools-spacemit`
（A100 核心，IME2 矩阵扩展）和 Ollama 运行大语言模型与嵌入模型的实测记录（2026 年 10 月），附一行修复补丁和全部数据。

1. **IME2 / A100 路径逐 token 解码结果错误。** `llama.cpp-tools-spacemit` 0.0.8 / 0.1.1 / 0.1.9 以及启用
   `-DGGML_CPU_RISCV64_SPACEMIT=ON` 的上游 llama.cpp 都受影响，生成文本中出现重复子词（"the the"、"cloudss"）。
   0.1.9 相对 F16 实测：Qwen2.5-7B Q4_K_M 平均 KLD 为 **0.286**（通用 CPU 路径为 0.046），top-1 一致率 76.7 %；
   其他版本相对 CPU 版本的偏差相同（KLD 0.214–0.215）。
2. **原因与修复。** `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:669` 中的 `float M[QLEN] = { -INFINITY };`
   只把 `M[0]` 初始化为 -∞。[一行修复](patches/0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch)
   后逐 token 结果与批处理一致（20 个 chunk：KLD **0.0667**），tg128 **5.12 t/s**。不改代码时可用 `-fa off`
   （prompt 约慢 11 %）。截至 2026-10-28，llama.cpp master `f5222f2` 中仍未修复。
3. **在 IME2 上用 Q4_0，不要用 Q4_K_M。** 后端对 Q4_K 的重排（`make_block_q4_1x32`，整数零点）带来
   6–13.5 % 的矩阵乘误差。Qwen2.5-7B **Q4_0**（`llama-quantize` 默认混合，output 为 Q6_K）加修复：KLD
   **0.032**，解码 **5.97 t/s**，prompt **57.5 t/s**。
4. **嵌入质量好，内存有问题。** Qwen3-Embedding-0.6B Q8_0 在厂商路径上与上游一致（余弦 ≥ 0.9989），本地单条
   **51 ms**，局域网客户端 **35 ms**。但 `llama-server --embedding` 处理不同长度的请求时不复用内存：
   `-c 4096` 涨到 **7.3 GB**，`-c 8192` 被 OOM killer 杀掉；只有短查询时 8 小时内存稳定（+53 MB）。
5. **A100 工作时不要打开 `cpufreq/boost`。** turbo 频率（A100 2.0 GHz）下结果随机出错（两次相同运行的平均
   KLD 分别为 124 和 0.089）；额定 1.8 GHz 下连续 60 次运行的统计结果完全相同。
6. **8 GB 板上一次只跑一个模型。** 7B 生成与嵌入服务同时运行导致系统卡死（无 swap，内存抖动）。厂商运行时的
   主线程留在 X100 上，会使同时在 X100 上运行的生成速度下降 77 %。

论坛讨论：[forum.spacemit.com/t/topic/1733](https://forum.spacemit.com/t/topic/1733)。
