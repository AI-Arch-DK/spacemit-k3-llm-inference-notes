# How the numbers were measured (methodology, limits, mistakes)

## Labels

- **[V]** measured live on the board (or on the LAN client talking to it);
- **[I]** inference or calculation from measured data;
- **[NV]** not verified by us: vendor documentation, reviews, or an open question.

## Test setup

| item | value |
|---|---|
| board | SpacemiT K3 Pico-ITX, part MPK3, 8 GB LPDDR5, 128 GB UFS, headless, active cooler; one sample |
| OS | Bianbu 4.0.1 (Ubuntu 26.04 "resolute" base); factory kernel 6.18.3-1.0.1.4 until 2026-10-21, then 6.18.3-1.0.8.4 |
| clocks | factory: `userspace` governor at X100 2.2 / A100 1.8 GHz; after the update: 2.0 / 1.8 GHz, `cpufreq/boost = 0`, unless stated |
| compiler | gcc 15.2.0 (Bianbu 15.2.0-16ubuntu1bb3) |
| llama.cpp | upstream `0a1ef30` (b11796, 2026-10-21) in three builds: generic CPU (RVV, run on X100 with `taskset -c 0-7`), with `-DGGML_CPU_RISCV64_SPACEMIT=ON`, and the same plus the [FA fix](../patches/) |
| vendor package | `llama.cpp-tools-spacemit` 0.1.9 (Bianbu); 0.0.8 and 0.1.1 from the Launchpad PPA `spacemit/k3`, unpacked into a separate directory and run with `LD_LIBRARY_PATH` |
| models | Qwen2.5-7B/3B/1.5B-Instruct Q4_K_M (published GGUFs; the 7B file sha256 `2bada8a7…`, 4,683,073,952 bytes); F16 reference `Qwen/Qwen2.5-7B-Instruct-GGUF` revision `bb5d59e0`; Q4_0 files quantized on the board from that F16; Qwen3-Embedding-0.6B Q8_0 (sha256 `06507c7b…`) and 4B Q4_K_M (sha256 `2b0cf8f1…`) |
| network | the board on 1GbE to one client PC through a USB Ethernet adapter |
| dates | 2026-10-21/22 (platform, speed, runtimes, firmware update), 2026-10-26/27 (correctness, fix, quantization, embeddings, reliability) |

## Accuracy: KL divergence instead of perplexity

`llama-perplexity` first saves reference logits (`--kl-divergence-base ref.kld`), then a second build or
model is compared against them (`--kl-divergence`). Reported: mean, median, 99th percentile and maximum
KLD, "Same top p" (the share of positions where both put the same token first — "same top-1" in these
notes) and PPL(Q)/PPL(base).

- **Text.** ~99 KB of Russian technical prose (internal documentation, not published), `-c 512`. 20
  chunks ≈ 5,100 scored tokens. Any text reproduces the effects; wikitext-2 should work too.
- **Two modes.** Batched: default `-ub 512`, as in prompt processing. Token-by-token: `-b 512 -ub 1`, the
  kernels text generation uses. The decode bug is only visible in the second mode.
- **References.** (a) upstream generic CPU build on X100, same model, in the matching mode; (b) the
  official F16 GGUF. The F16 model is 15.2 GB; on the 8 GB board it runs through mmap without swap
  (peak RSS 7.6 GB, 125 s per 512-token chunk at 2.0 GHz, 42 minutes for 20 chunks), reading the weights
  from UFS once per batch of 4 chunks. F16 and Q4_K_M share the vocabulary byte for byte (152,064 tokens,
  same BOS/EOS).
- **Controls**, all on the same text: upstream `-t 4` vs `-t 8` → 0.000000; vendor run vs vendor run →
  0.000000; upstream with vs without weight repack → 0.0056; upstream batched vs token-by-token → 0.0046.
  These set the noise floor; everything above ~0.01 is a real difference.
- **Perplexity is a poor judge here**: on this text Q4_K_M has lower PPL than F16 (13.58 vs 13.69).
  Decisions are made on KLD.
- **4-chunk and 20-chunk numbers are not mixed** in one comparison. Where a 4-chunk value is quoted from a
  20-chunk run, it is the cumulative value at chunk 4 of that run and is marked so.
- Every KLD number in these notes comes from a `llama-perplexity` report; the table
  [`data/kld_results.csv`](../data/kld_results.csv) was rebuilt from those reports.

## Upstream status check (2026-10-28)

Through the GitHub API: the head of `ggml-org/llama.cpp` master was `f5222f2` (committed 2026-10-28); its
`ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp` is byte-identical to the one at `0a1ef30` and still has the
faulty line 669; the patch applies to it with `git apply --check`; a search for "spacemit" in the issues
and pull requests of `ggml-org/llama.cpp` found no report of this bug (it found #29261 and #30284, both
about default thread counts). The SpacemiT forum topic was read through its public JSON.

## Greedy generation check

8 real-world prompts (one-word classification, extraction to JSON, error extraction, summarisation,
translation, short explanation, SQL advice, comparison of options) in the Qwen2.5 chat template,
`llama-server` `/completion`, temperature 0, `top_k` 1, each build twice. The prompt token counts were
checked to be equal across builds, so the input is identical. "Duplication artifacts" were counted by
hand in the answers.

## Speed

`llama-bench -p 512 -n 128`, `-r 3` or five separate runs (median), one run per configuration where
marked. Before and after each step: clocks, temperatures of 7 zones, free memory (`temps.jsonl` in the raw
data). The vendor runtime picks its cores itself; the generic CPU build was pinned to X100
(`taskset -c 0-7 … -t 8`). Context-depth runs use `-d` (this llama-bench version has no `-c`).

## Embeddings

`llama-server --embedding --pooling last`, `/v1/embeddings`. Quality: cosine between vectors of the same
text from different builds, and retrieval overlap (top-10 neighbours of 28 queries among 139 paragraphs).
Latency from a LAN client: only in windows where the client's CPU was < 15 % busy and its local model
server was idle; RSS of the server from `/proc/<pid>/status` (`VmRSS`, `VmHWM`).

## Limits

- **One board, 8 GB variant.** The 16/32 GB variants may behave differently, especially in memory limits.
- No wattmeter: power is [NV]. M.2 was empty (no NVMe, no accelerator), 10GbE had no second end, no monitor.
- The Bianbu archive was unreachable on 2026-10-26, so vendor versions 0.0.9, 0.1.3, 0.1.6 and 0.1.7 were
  not tested.
- X100 at 2.4 GHz with A100 capped at 1.8 GHz was not tested; neither was the upstream IME2 build in
  embedding mode for memory growth, `-t 7` for a generator next to the embedder, a 4B embedder in
  Q8_0/Q4_0, or a series of software reboots.

## Our mistakes, and what was redone

Listed because they shaped what the data can and cannot show:

1. **A100 runs through cgroups.** The first A100 measurements used `systemd-run -p AllowedCPUs=8-15`, which
   bypasses HMP (kernel warning `hmp.c:204`). All of them were discarded and redone through
   `/proc/set_ai_thread`.
2. **A broken log column.** In the 30-minute generation run the tok/s column was `NaN` (an escaping bug in
   the loop, not in generation). The thermal log stayed valid; tok/s was re-measured in a separate
   10-minute run.
3. **Wrong reading of the firmware update.** We first called the new frequency limits a lower speed bin;
   they are `turbo-mode` OPPs behind `cpufreq/boost`. Corrected on 2026-10-26.
4. **Speed numbers spoiled by a background download.** Two boost-speed runs were taken while the F16 model
   was being copied over the network on X100; they are not used.
5. **OOM in a 20-chunk run.** A quantization job was started in parallel; the run was killed and repeated
   on 4 chunks, and the 20-chunk number was taken in a later, guarded run.
6. **A hang caused by the test plan.** 7B generation next to a grown embedding server, without a memory
   guard; see [reliability log](clocks-boost-firmware-reliability.md#reliability-log-resets-hangs-reboots).
7. **Wrong input texts in the long-text test.** Synthetic texts were cut from a file that was being edited
   at the same time, so a re-run got a different text (a false cosine of 0.9845). Texts are now taken from
   a fixed revision, and the affected rows were re-run; one file was also saved without its last two
   records and was recomputed.
8. **Misleading exit codes** in our orchestration logs: `echo "$(ts) … rc=$?"` reports the exit code of
   `$(ts)`, not of the step. The on-board step chain logged correct codes; conclusions do not depend on
   the broken ones.
9. **A kernel stand bug.** The first stand run passed 256 columns to a kernel that takes at most 32 per call
   and got 93 % "error"; fixed before any number was used.
