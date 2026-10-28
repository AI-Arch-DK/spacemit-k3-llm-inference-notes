# SpacemiT K3 Pico-ITX platform notes: X100 vs A100 (HMP, `/proc/set_ai_thread`), memory bandwidth, storage, network, runtimes on riscv64

What a developer needs to know about the K3 before running LLMs on it, measured on one K3 Pico-ITX (8 GB)
with Bianbu 4.0.1, October 21–27, 2026. Labels: **[V]** measured, **[I]** inference, **[NV]** not
verified (vendor documentation or third-party).

- [Board inventory](#board-inventory)
- [Two kinds of cores: how the A100 cluster is reached (HMP)](#two-kinds-of-cores-how-the-a100-cluster-is-reached-hmp)
- [The vendor AI runtime: TCM, IME2, `/dev/tcm_sync_mem`](#the-vendor-ai-runtime-tcm-ime2-devtcm_sync_mem)
- [CPU throughput per cluster](#cpu-throughput-per-cluster)
- [Memory bandwidth and latency](#memory-bandwidth-and-latency)
- [Where to run what: background load and contention](#where-to-run-what-background-load-and-contention)
- [Context length on 8 GB](#context-length-on-8-gb)
- [Runtimes on riscv64: llama.cpp, Ollama, Vulkan, Node, Docker](#runtimes-on-riscv64-llamacpp-ollama-vulkan-node-docker)
- [Storage and network](#storage-and-network)
- [Which OS](#which-os)

## Board inventory

| item | K3 Pico-ITX, part MPK3, as tested | label |
|---|---|---|
| CPU | 16 RISC-V cores in 4 clusters of 4: **cpu0–7 X100** (VLEN 256, `vlenb` 32; `rv64imafdcvh` + Zb*, Zfh/Zvfh, Zve64d, Zvk*, hypervisor), **cpu8–15 A100** (VLEN 1024, `vlenb` 128, no `h`, DT property `cpu-ai`); RVA23 profile per vendor | [V] / [NV] |
| caches | L1d/L1i 64 KB per core; L2 4 MiB per X100 cluster (`lscpu`: 10 MiB in 4 instances → 4+4+1+1) | [V] / [I] |
| memory | 8 GB LPDDR5 soldered (7.97 GiB visible), no swap; LPDDR5-6400, 2 × 32 bit → 51.2 GB/s theoretical | [V] / [NV] |
| storage | UFS 2.2, Kingston TY7B-128 (119 GB); 8 MB SPI NOR for the boot loader; no eMMC | [V] |
| PCIe | RC0 Gen3 ×4 → M.2 M-key 2280 (empty); RC2 Gen3 ×2 → RTL8127 10GbE SFP+ (8 GT/s ×2); RC4 Gen3 ×1 → RTL8852BE Wi-Fi 6 | [V] `lspci -vv` |
| network | 1GbE RJ45 (SoC GMAC), 10GbE SFP+ (not tested, no second end), Wi-Fi 6 + BT | [V] |
| USB | three USB 3.0 buses (5 Gbit/s) and four USB 2.0; no USB4/Thunderbolt, so no eGPU or PCIe accelerators over USB | [V] |
| GPU | Imagination PowerVR B-Series BXM-4-64 MC1; Vulkan 1.3.277 ICD, OpenCL 3.0 ICD | [V] |
| virtualisation | `/dev/kvm` present (X100 only); **no IOMMU** (no PCI passthrough) | [V] |
| ESOS | `bianbu-esos` — an RTOS on a separate remote processor (`remoteproc`, rpmsg `adma-service`), not on cpu8–15 | [V] |

## Two kinds of cores: how the A100 cluster is reached (HMP)

The X100 and A100 cores have different vector register sizes (VLEN 256 vs 1024), so a thread must never
migrate between clusters. The SpacemiT kernel (`CONFIG_SPACEMIT_HMP`, described in the vendor's "K3 HMP
User Guide") therefore splits the CPUs into a regular mask and an AI mask [V]:

- every process is REGULAR by default and runs **only on X100**: `nproc` = 8, and "16 threads" without
  more means 8 X100 cores with two threads each;
- `sched_setaffinity` across the boundary fails with `EINVAL`; IRQs and KVM run only on X100;
- the official entry to the A100 cluster is **`/proc/set_ai_thread`** — write a PID, or `0` for the
  calling thread, and that thread becomes an AI thread that lives only on cpu8–15. The file is
  `--w--w--w-`: any user can write it;
- children of an AI thread stay AI threads (`bash` after `fork` and a Python child both had
  `Cpus_allowed: 8-15`);
- **do not** move processes onto cpu8–15 with cgroups (`systemd-run -p AllowedCPUs=8-15`): it bypasses HMP,
  the kernel prints `WARNING ... drivers/soc/spacemit/hmp.c:204 _set_cpumask`, and we discarded every
  number measured that way.

A minimal wrapper to run a program on the A100 cluster ([`reproduce/run-on-a100.sh`](../reproduce/run-on-a100.sh)):

```bash
#!/bin/bash
echo 0 > /proc/set_ai_thread || exit 1   # this shell becomes an AI thread (A100 only)
exec "$@"                                # the program inherits it
```

The vendor llama.cpp does this by itself: its **main thread stays on X100** (`Cpus_allowed 0-7`) and the 8
worker threads are moved to cpu8–15; `-t` is ignored (default, `-t 8` and `-t 16` gave the same speed).
The kernel then logs `hmp_set_ai_thread: pid:N(llama-server), vector has been enabled already!!!` for each
worker: the runtime marks threads as AI after they have already used the vector unit, which the HMP guide
says not to do [V].

## The vendor AI runtime: TCM, IME2, `/dev/tcm_sync_mem`

`llama.cpp-tools-spacemit` 0.1.9 depends on `spacemit-runtime` 0.6.2, `spacemit-tcm` 3.0.1 and
`spacemit-onnxruntime` 2.0.7. Every start prints (vendor package and upstream with
`-DGGML_CPU_RISCV64_SPACEMIT=ON` alike) [V]:

```
tcm is available, blk_size 393216, blk_num 8
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064, exclude_main_thread: 0,
  use_ime1: 0, use_ime2: 1, mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
CPU_RISCV64_SPACEMIT: alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
failed to allocate init_barrier from shared mem, falling back to heap
```

So the "60 TOPS" of the K3 is not a separate NPU: it is the A100 cluster with the IME2 matrix extension
and 3 MiB of TCM, used by the vendor runtime. `/dev/tcm_sync_mem` is missing before and after the
October update; the TCM blocks are still allocated (`spacemit-tcm-smi`: 8/8 free at idle). Whether the
missing device affects speed is an open question to SpacemiT.

The vendor DDR traffic monitor `ddr_bw` (package `ddr-bw`) does **not see the A100 traffic**: during vendor
generation, ports 0–4 and `all` showed ~0 read, while upstream generation on X100 showed 10–14 GB/s. It
also buffers stdout (~4 KB) when not on a terminal; `script -qfc "ddr_bw -p N" file` gives line output,
`stdbuf -oL` does not [V].

## CPU throughput per cluster

`sysbench cpu --time=60` (prime numbers), 2.2 / 1.8 GHz, factory firmware [V]:

| configuration | threads | events/s | per thread |
|---|---|---|---|
| X100, 1 thread | 1 | 2,313 | 2,313 |
| X100 ×8 (`taskset 0-7`) | 8 | 18,403 | 2,300 |
| "all 16" without AI marking (= 8 X100) | 16 | 18,333 | — |
| A100 ×8 as AI threads | 8 | 8,460 | 1,058 |

An A100 core does **46 %** of an X100 core on scalar integer code; the clock difference (1.8 vs 2.2 GHz)
accounts for a factor of 0.82, so per GHz an A100 core does ~56 % of an X100 core [I]. Run Python, Node, SQLite and other general services on X100.

## Memory bandwidth and latency

`bench_membw` (OpenMP, `-O3 -march=rv64gcv_zba_zbb_zbs -ffast-math -fopenmp`, 256 MiB arrays, best of 10),
2.2 / 1.8 GHz, GB/s [V]:

| configuration | threads | READ | COPY | TRIAD |
|---|---|---|---|---|
| X100 | 1 | 8.64 | 12.56 | 10.46 |
| X100 | 2 | 15.47 | 20.36 | 17.29 |
| X100 | 4 | 16.45 | 19.40 | 19.07 |
| X100 | 8 | **20.70** | 19.19 | 19.49 |
| A100 (AI threads, `OMP_PLACES=cores`) | 1 | 9.66 | 18.01 | 14.10 |
| A100 | 2 | 19.13 | 28.78 | 26.92 |
| A100 | **4** | **31.88** | 28.98 | 26.15 |
| A100 | 8 | 23.43 | 22.06 | 22.80 |
| **both clusters at once** (two processes) | 8 + 8 | 20.00 + 16.89 = **36.9** | 18.92 + 15.92 | 19.15 + 16.28 |

- The ~21 GB/s ceiling of X100 is a property of the cluster path, not of DRAM: four A100 cores alone read
  31.9 GB/s (62 % of the 51.2 GB/s theoretical peak), both clusters together 36.9 (72 %) [I].
- On A100, 8 threads are slower than 4 (23.4 vs 31.9).
- X100 sustained, 8 minutes: READ min/median/max 20.47 / 20.79 / 20.95 GB/s, no bimodality.
- At 2.4 / 2.0 GHz READ on X100 is 20.94 — bandwidth does not depend on the core clock.
- Latency (`tinymembench`, X100, single random read): 1 MiB 10.1 ns (L2), 4 MiB 21.0 ns, 16 MiB 119.5 ns,
  64 MiB **151.3 ns** (dual random read 167.5 ns). Single-thread memcpy 6.14 GB/s, memset 17.6 GB/s.

Effective bandwidth during 7B Q4_K_M decode (tok/s × 4.68 GB per token) [I]: vendor IME2 path 26.5 GB/s
(5.65 tok/s), upstream on X100 16.0 GB/s (3.41 tok/s). Smaller models drop below that on X100 (3B 11.4,
1.5B 9.6 GB/s; those runs used `-t 16`, two threads per core, which also costs speed): X100 becomes
compute-bound [I]; the vendor path stays at 22–27 GB/s on every model size
(3B 23.9, 1.5B 22.2).

## Where to run what: background load and contention

**Background on X100, LLM on A100 vs on X100.** 7B Q4_K_M tg128 (median of 3) under synthetic background
on the X100 cores, 2.2 / 1.8 GHz [V]:

| background on X100 | vendor path (A100) | upstream (X100 ×8) |
|---|---|---|
| none | 5.51 | 3.40 |
| desktop-like load (2 workers × 75 %) | 5.48 (−0.5 %) | 1.88 (−45 %) |
| service load (an embedding server on 4 X100 threads + a search query every 2 s + an SQLite FTS5 loop) | 5.32 (−3.5 %) | 2.52 (−26 %) |
| all of the above + a busy Node.js server | **5.20 (−5.6 %)** | **1.71 (−50 %)** |

Prompt processing on the vendor path did not change at all (41.6–42.6 tok/s). This is the value of the
HMP split: a generator on A100 is nearly immune to whatever runs on X100.

**Services on A100: no.** The same service workloads on X100 vs as AI threads on A100 [V]:

| workload | X100 | A100 | A100 / X100 |
|---|---|---|---|
| Python + SQLite FTS5, 200 queries | 1.10 ms/query | 2.24 | 2.0× slower |
| Python `json.loads` of 1,184 vectors | 1,383 ms | 2,875 | 2.1× slower |
| Node.js HTTP server, 200 requests | p50 2.1 / p95 3.6 ms | p50 6.0 / p95 9.1 | 2.9× slower |
| vendor 7B tg128 while that background runs | 5.12 (−7 %) | **2.49 (−55 %)** | — |

During generation the vendor runtime occupies the whole A100 cluster (9 threads); anything else placed
there slows both. Layout that works: **A100 only for the vendor LLM / embedder, everything else on X100.**

**Memory controller contention.** Upstream 7B generation on X100 while A100 ran a memory stream (plus UFS
reads and iperf3): generation 3.41 → **1.00–1.19 tok/s (−68 %)**, A100 stream 31.9 → 27.8 GB/s (−13 %),
UFS and network unaffected. One memory controller serves both clusters, and the X100 cluster with its
lower own bandwidth loses [V].

## Context length on 8 GB

Vendor path, 7B Q4_K_M, `llama-bench -d` (2.2 / 1.8 GHz) [V]: tg128 at depth 2048 **4.24 tok/s** (−25 %),
at depth 8192 **2.45 tok/s** (−57 %, 478 s per run, **max RSS 7.40 GB of 7.93**). Upstream on X100 at
depth 2048: 2.96 tok/s (−13 %). For a 7B on the 8 GB board keep the context at 4096 or below, or use the
16/32 GB variants. Models of the 20–30B class do not fit in 8 GB at all.

## Runtimes on riscv64: llama.cpp, Ollama, Vulkan, Node, Docker

State on 2026-10-21; the IME2 build on 2026-10-26 [V]:

| runtime | result |
|---|---|
| **upstream llama.cpp** (`0a1ef30`) | builds natively with `-DGGML_NATIVE=ON -DGGML_RVV=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZICBOP=ON`, gcc 15.2. On X100 7B Q4_K_M 10.0 / 3.41 tok/s (pp512 / tg128). On A100 as AI threads the generic RVV path is unusable: 0.5 / 0.46 tok/s |
| **upstream llama.cpp + SpacemiT IME2 backend** | `-DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_CPU_REPACK=OFF` + RVV flags, native gcc 15.2 without the vendor toolchain, 12 min on X100; compiler checks for IME1/IME2 pass, `-march` gets `_xsmtvdotii`. KLD results identical to the vendor package to the 6th digit (including the [decode bug](decode-bug-flash-attention-ime2.md)); speed 38.48 / 5.08 tok/s vs 42.4 / 5.06 for the package (pp512 / tg128, 2.0 / 1.8 GHz) |
| **`llama.cpp-tools-spacemit`** (Bianbu) | `/usr/bin/llama-*`; A100 + IME2 + TCM; 7B Q4_K_M 42.9 / 5.65 tok/s (factory clocks) |
| **Ollama** | no riscv64 release; **builds from source** (commit f580b54, Go 1.26 from the Bianbu archive). The cmake step puts the runner `llama-server` and `libggml-cpu-riscv64_v.so` into `build/llama-server-local/bin/`, but `ollama serve` looks in `lib/ollama/` next to the binary ("llama-server binary not found") — copy it there and it works. **Set `num_thread` to 8**: Ollama counts 16 CPUs, and 16 threads on 8 X100 cores give **0.16–0.22 tok/s**; with `num_thread: 8` → **2.65 tok/s**. The runner variant `riscv64_v` is built without zfh/zvfh/zicbop: −22 % decode, −53 % prompt against upstream `llama-server` (3.41 / 9.4). No access to A100/IME2 |
| **Vulkan on PowerVR** | llama.cpp with `GGML_VULKAN=ON` builds (needs `spirv-headers`, `glslang-tools`, `spirv-tools`, `libshaderc-dev`); `--list-devices` shows "PowerVR B-Series BXM-4-64 MC1, uma: 1, fp16: 1, int dot: 0, matrix cores: none"; `llama-bench -dev Vulkan0` aborts with `vk::Device::createComputePipeline: ErrorUnknown` at `-ngl 99` and `-ngl 20`. GPU path for LLMs closed; no ggml OpenCL backend for PowerVR |
| Node.js | `nodejs` 22.22.1 and npm 9.2.0 from the Bianbu archive work |
| Claude Code CLI 2.x | `npm i -g @anthropic-ai/claude-code` installs, but `claude` fails: "claude native binary not installed … platform-native optional dependency was not downloaded" — no riscv64 build |
| Docker | `docker.io` 29.1.3 from the archive works (`riscv64/alpine`) |
| SpacemiT AI packages | `llama.cpp-tools-spacemit`, `spacemit-onnxruntime` / `python3-spacemit-ort`, `spacemit-runtime`, `spacemit-tcm`, `openblas-spacemit`, `openvml-spacemit`, `eigen-spacemit`, `opencv-spacemit`, `ddr-bw`, `mpp2` |

Ollama with `num_thread: 8` (or `OLLAMA_NUM_THREADS`) works for small jobs, but it cannot use the A100
cluster: for LLMs on the K3, use `llama-server` with the IME2 backend and the [fix](decode-bug-flash-attention-ime2.md).

## Storage and network

UFS 2.2 (`fio`, 60 s, `direct=1`, 4 GB file) [V]:

| test | bandwidth | IOPS | latency mean / p99 |
|---|---|---|---|
| seq read 1M QD8 | 586 MiB/s | 586 | 13.7 / 15.7 ms |
| seq write 1M QD8 | 165 MiB/s | 165 | 48.6 / 63.2 ms |
| rand read 4k QD32 | 26 MiB/s | 6,706 | 4.77 / 17.7 ms |
| rand write 4k QD32 | 115 MiB/s | 29,391 | 1.09 / 4.69 ms |

Loading a 4.7 GB model from UFS takes ~8–10 s; keep the model resident. 1GbE (`iperf3`, 20 s) [V]:
941 Mbit/s TCP in both directions with 0 retransmits, at 2–31 % CPU of the board; UDP 1400 B board → client
950 Mbit/s.

## Which OS

Criterion: the board is "complete" only with the SpacemiT kernel (HMP + TCM) and the vendor AI stack.
State of the archives on 2026-10-21 [V, apt / Launchpad]:

| OS | kernel | SpacemiT AI stack | verdict |
|---|---|---|---|
| **Bianbu 4.0.1** (factory; Ubuntu 26.04 "resolute" base) | 6.18.3 with HMP and TCM | full: `spacemit-runtime` 0.6.2, `spacemit-tcm` 3.0.1, `llama.cpp-tools-spacemit` 0.1.9, `spacemit-onnxruntime` 2.0.7 | most complete; vendor repository, not a Canonical LTS channel |
| Ubuntu 26.04 + PPA `spacemit/k3` (`spacemit-com/K3-Ubuntu-Images`) | `linux-image-spacemit-generic` 6.18.3 (same SpacemiT tree) | present but a generation older: `llama.cpp-tools-spacemit` 0.1.1, ORT 2.0.3 with the runtime inside, `spacemit-tcm` 3.0.0 | equivalent, lagging versions; boots via EDK2/GRUB |
| Ubuntu 26.04 + PPA `ubuntu-risc-v-team/k3` | `linux-image-k3-6.18` (SpacemiT-derived; HMP [NV]) | none | 8 X100 cores only |
| Fedora / Debian / Armbian with a mainline kernel | no HMP / TCM | none | all 16 cores visible to the scheduler → unsafe vector migration |

We stayed on Bianbu; the Ubuntu image was not tested.
