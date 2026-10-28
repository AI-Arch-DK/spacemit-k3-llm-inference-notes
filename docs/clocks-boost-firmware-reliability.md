# Is cpufreq boost (turbo) safe on SpacemiT K3? Firmware update 1.0.8, OPP tables, A100 errors, resets

**Short answer.** Do not enable `cpufreq/boost` for work on the A100 cores. After the October 2026
firmware/kernel update the top clocks (X100 2.2–2.4 GHz, A100 1.85–2.0 GHz) became `turbo-mode` OPPs
behind `/sys/devices/system/cpu/cpufreq/boost`. With boost on, the A100 / IME2 path produced **randomly
corrupted results** (mean KLD 124 in one run, 0.089 in an identical second run); at the nominal 1.8 GHz,
all **60** consecutive runs gave the same result statistics. The X100 cores at 2.4 GHz computed correctly. A
30-minute `stress-ng` at boost passed — it does not check results.

Labels: **[V]** measured, **[I]** inference, **[NV]** not verified. One board sample (K3 Pico-ITX, 8 GB).

- [Clocks as shipped and after the update](#clocks-as-shipped-and-after-the-update)
- [OPP tables and voltages](#opp-tables-and-voltages)
- [A100 errors at turbo clocks](#a100-errors-at-turbo-clocks)
- [A100 at the nominal 1.8 GHz is deterministic](#a100-at-the-nominal-18-ghz-is-deterministic)
- [Speed before and after the update](#speed-before-and-after-the-update)
- [Thermals](#thermals)
- [Reliability log: resets, hangs, reboots](#reliability-log-resets-hangs-reboots)
- [Package archive](#package-archive)

## Clocks as shipped and after the update

| | factory image (2026-10-21) | after `apt full-upgrade` (2026-10-21/22) |
|---|---|---|
| kernel | `linux-image-6.18.3-generic` 6.18.3-1.0.1.4 | 6.18.3-1.0.8.4 |
| U-Boot / OpenSBI / ESOS | `u-boot-spacemit` 1.0.1 / `opensbi-spacemit` 1.0.1 / `bianbu-esos` 1.0.1 | 1.0.8 / 1.0.8 / 1.0.8 |
| EC | `spacemit-ec-firmware` 0.0.19 | 1:0.0.32 |
| other | DTB `spacemit-flash-dtbs` 1.3.6; boot from SPI NOR (`boot_mode=nor`), no UEFI | `linux-firmware-spacemit` 0.0.14 |
| cpufreq policies | cpu0–7 (X100) 614 MHz–2.4 GHz; cpu8–15 (A100) 614 MHz–2.0 GHz | `cpuinfo_max_freq` 2.15 GHz (X100) / 1.8 GHz (A100); frequency list 19 → 16 steps (X100), 14 → 11 (A100) |
| governor as shipped | `userspace`, fixed at **2.2 / 1.8 GHz** | `userspace`, **2.0 / 1.8 GHz** |
| `cpufreq/boost` | — | 0 by default; `scaling_boost_frequencies`: X100 2.2/2.3/2.4 GHz, A100 1.85/1.9/2.0 GHz |

The AI stack was the same before and after: `llama.cpp-tools-spacemit` 0.1.9, `spacemit-runtime` 0.6.2,
`spacemit-tcm` 3.0.1, `spacemit-onnxruntime` 2.0.7. The update itself (102 packages) finished with `rc=0`, but the reboot that applied it did not complete
(link up, no frames); a power cycle brought the board up on the new kernel [V].

We first read the shorter frequency list as a lower speed bin chosen for this chip. That was wrong: the
top steps are still in the device tree, marked `turbo-mode`, and come back with `boost=1` [V]. `boost` is a
global switch; capping the A100 policy (`cpu8` `scaling_max_freq` = 1.8 GHz) while boost is on would give
X100 2.4 GHz with A100 at nominal clocks [I, not tested].

## OPP tables and voltages

From the device tree (`/sys/firmware/fdt` and the `.dtb` of both kernel packages;
[`data/firmware_opp_tables.txt`](../data/firmware_opp_tables.txt)), X100 cluster, first `opp-microvolt`
value [V]:

| X100 frequency | factory DT | new DT, `opp_table0_x100` | new DT, `opp_table1_x100` |
|---|---|---|---|
| 614 MHz – 1.0 GHz | 800 mV | 950 mV | 910 mV |
| 1.1 – 1.5 GHz | 850 mV | 950 mV | 910 mV |
| 1.6 – 2.15 GHz | 880 mV | 950 mV | 910 mV |
| 2.2 GHz | 880 mV | 950 mV, turbo | 950 mV, turbo |
| 2.3 GHz | 880 mV | 1000 mV, turbo | 1000 mV, turbo |
| 2.4 GHz | 1000 mV | 1000 mV, turbo | 1000 mV, turbo |

The A100 table has no voltages in the DT; its 1.85/1.9/2.0 GHz steps are marked `turbo-mode`. The new
kernel picks the X100 table by PMIC (`Spacemit K3: pmic-compatible="au4562", using X100 OPP table0` in
`dmesg`; the old kernel has no such line). `spacemit-ec-boot-update` updates the EC only upwards and only
with `u-boot-spacemit` ≥ 1.0.4 ("SPL voltage fix") [V, package scripts].

[I] The factory firmware ran X100 at 2.2 GHz at 880 mV; the new one asks for 950 mV at the same clock.
One spontaneous reset at 2.2 GHz under sustained memory load on the factory firmware
([below](#reliability-log-resets-hangs-reboots)) is consistent with that, but it is not proven.

## A100 errors at turbo clocks

`llama-perplexity` with KL divergence against fixed reference logits is a far better stability test than
`stress-ng`, because it checks every output. Vendor `llama.cpp-tools-spacemit` 0.1.9, Qwen2.5-7B Q4_K_M,
batched, 2 chunks, against upstream CPU token-by-token logits ([`data/kld_results.csv`](../data/kld_results.csv)) [V]:

| run | clocks X100 / A100 | mean KLD | max KLD | same top-1 | PPL ratio |
|---|---|---|---|---|---|
| vendor, run 1 | 2.4 / 2.0 GHz (boost) | **124.4** | **63,058** | 91.18 % | overflow (~10^55) |
| vendor, run 2 (identical command) | 2.4 / 2.0 GHz (boost) | **0.089** | **16.3** | 91.37 % | 1.071 |
| vendor | 2.0 / 1.8 GHz (nominal) | 0.029 | 0.38 | 91.57 % | 1.004 |
| upstream generic CPU on X100 | 2.4 / 2.0 GHz (boost) | 0.0046 | 0.037 | 96.67 % | 1.005 |

Different garbage in different places on each boosted run; clean and reproducible at nominal clocks;
X100 at 2.4 GHz clean. In the same session, 30 minutes of a memory stream on X100 ×8 at 2.4 GHz plus
`stress-ng` on A100 ×8 at 2.0 GHz ran without reset, at ≤ 56 °C, clocks held, `stress-ng` reported 0
errors [V] — it does not verify results, so it does not catch this.

## A100 at the nominal 1.8 GHz is deterministic

Vendor build, batched, 4 chunks, **60 consecutive runs over 2.5 hours**, X100/A100 at 2.0/1.8 GHz the whole
time, zone temperatures 47–49 °C [V]: all 60 runs produced the same statistics against logits saved by
the same build a day earlier — mean KLD 0.000000, 99.9 % 0.000040, max 0.000048, same top-1 100 %.
The small constant residual against the earlier run is the same in all 60 runs, i.e. a fixed difference
from that run, not a random error [NV, cause not investigated].

## Speed before and after the update

Qwen2.5-7B-Instruct Q4_K_M, quiet board [V]:

| | before (2.2 / 1.8 GHz, 2026-10-21) | after, 2.0 / 1.8 GHz | after, boost 2.4 / 2.0 GHz |
|---|---|---|---|
| X100 memory READ, `bench_membw` 8 threads | 20.8 GB/s | 19.8–20.0 GB/s | 20.1–20.4 GB/s |
| upstream generic CPU on X100, tg128 | 3.41 tok/s | 3.11 | 3.30 |
| vendor 0.1.9 on A100, tg128 | 5.65 tok/s | 5.06 | 5.10 |
| vendor 0.1.9 on A100, pp512 | 42.9 tok/s | 42.4 | 46.5 |

Upstream on X100 scales with the X100 clock. Vendor decode barely reacts to clocks, and the A100 ran at
1.8 GHz both before and after the update, so its −10 % after the update is **not** explained by core
clocks or memory bandwidth [NV — open question to SpacemiT]. Prompt processing scales with frequency.

## Thermals

Active cooler with a PWM fan (8 steps), EC `cros_ec`; seven thermal zones (`top`, `vpu`, `gpu`,
`cluster0–3`) [V]:

| load | duration | result |
|---|---|---|
| `stress-ng --cpu 8` on X100 + 8 on A100 (as AI threads), 2.2 / 1.8 GHz | 30 min | cluster0 max 62 °C, A100 cluster max 59 °C; clocks unchanged; throttling state 0; fan step 3 of 8 (2,528–2,580 rpm); X100 868 / A100 551 bogo ops/s |
| vendor 7B generation in a loop, 2.2 / 1.8 GHz | 30 min | all zones 52–57 °C; no throttling; tok/s first → last: 5.58 → 5.62 (10-min repeat) |
| X100 memory stream at 2.4 GHz + `stress-ng` A100 at 2.0 GHz (boost) | 30 min | ≤ 56 °C, no throttling |
| embedding service, 0.6B, one query per 180 s | 8 h | 42–45 °C |

Without a wattmeter on the bench, power draw is **[NV]** (reviews quote ~7 W idle, ~22 W CPU load,
~39 W max — low confidence).

## Reliability log: resets, hangs, reboots

Everything that went wrong with the board during a week of testing, including our own mistakes [V]:

1. **Spontaneous reset, factory firmware (2026-10-21).** During a sustained memory-bandwidth run (8 threads
   on X100 at 2.2 GHz, 8th minute). The previous boot's journal ends on ordinary lines without service
   shutdown; the kernel reports "not because of SoC watchdog reset"; pstore empty; a second before: top
   58 / cluster0 60 / cluster1 56 °C (trip points 85–115 °C). A 5-minute canary with the same load did not reproduce it.
   Thermal and software causes are unlikely; power (unknown PSU type) or external interference remain.
2. **Reboot after the firmware update did not complete (2026-10-21)** — power cycle on 2026-10-22 (above).
3. **Hang under memory pressure (2026-10-27)** — 7B generation started next to a grown embedding server, no
   swap: thrashing, ping alive, ssh dead, no OOM-kill line. **Our test design error** (that step had no
   memory guard). Power cycle; see [embeddings](embeddings-server.md#running-an-embedder-and-a-generator-on-one-8-gb-board).
4. **OOM kills (2026-10-26/27)**: a 20-chunk KLD run killed because we ran quantization in parallel (our
   error, repeated on 4 chunks); a vendor embedding server with `-c 8192` killed after growing to 7.5 GB.
5. **Two power interruptions of the test bench (2026-10-27)**: the board lost power together with its
   network link to the client (external cause). The board booted again (whether by itself or by hand
   cannot be told from the journal) and resumed service.
6. **Software reboot on the updated system works (2026-10-27)**: `systemctl reboot` → ping lost after
   3.2 s → back after 24.4 s → ssh after **25.5 s**; clean shutdown in the previous boot's journal;
   `boost=0` and `userspace` 2.0/1.8 GHz preserved. One attempt, not a series.
7. **No `/dev/watchdog`**: the device tree has a watchdog node, but the driver is not enabled in the Bianbu
   kernel — relevant for a 24/7 node.

## Package archive

On 2026-10-26 `archive.spacemit.com` resolved through its CDN (`archive.spacemit.com.w.cdngslb.com`) to
`offline.specialcdnstatus.com` → `169.254.254.254`, i.e. the Bianbu repository was unreachable from every
network we tried, direct and through a VPN [V]. Older `llama.cpp-tools-spacemit` builds were taken from
the Launchpad PPA `spacemit/k3` instead (sha256 checked against the Launchpad API).

The archive index on 2026-10-22 had `linux-image` back to 1.0.1.4, but `u-boot-spacemit` and
`opensbi-spacemit` only from 1.0.4 and the EC only from 0.0.26, and the EC can only be downgraded by hand:
**a rollback to the factory firmware is not possible from the vendor archive** [V].
