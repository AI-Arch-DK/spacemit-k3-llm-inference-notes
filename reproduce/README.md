# Reproduce on a SpacemiT K3

Commands to repeat the main measurements on your own K3 board (Bianbu 4.0.x). The scripts are
generalised versions of what ran on our board (paths and names changed); they were not re-run on the
board after that change, so treat them as a recipe and check the output.

- [0. Prerequisites](#0-prerequisites)
- [1. Decode defect: KLD token-by-token vs batched](#1-decode-defect-kld-token-by-token-vs-batched)
- [2. Speed](#2-speed)
- [3. Accuracy against F16 and the Q4_0 file](#3-accuracy-against-f16-and-the-q4_0-file)
- [4. Boost / turbo correctness check](#4-boost--turbo-correctness-check)
- [5. Embedding server memory growth](#5-embedding-server-memory-growth)
- [6. IME2 kernel stand](#6-ime2-kernel-stand)

## 0. Prerequisites

```bash
sudo apt install build-essential cmake git python3
sudo apt install llama.cpp-tools-spacemit        # optional: the vendor build, /usr/bin/llama-*
bash build-llama-cpp.sh                          # upstream 0a1ef30: build-cpu, build-ime, and build-fafix with the fix
cat /sys/devices/system/cpu/cpufreq/boost        # keep 0 (see docs/clocks-boost-firmware-reliability.md)
```

Models (any Qwen2.5-Instruct GGUF reproduces the effects): `Qwen/Qwen2.5-7B-Instruct-GGUF` (F16 split
files and Q4_K_M), `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0). Text: any ~100 KB plain-text file.

Run the generic CPU build on X100 explicitly (`taskset -c 0-7 … -t 8`); the IME2 builds move their workers
to the A100 cores themselves. To run anything else on A100, use [`run-on-a100.sh`](run-on-a100.sh).

## 1. Decode defect: KLD token-by-token vs batched

```bash
MODEL=qwen2.5-7b-instruct-q4_k_m.gguf TEXT=text.txt \
CPU=llama.cpp/build-cpu/bin IME=llama.cpp/build-ime/bin bash kld-check.sh
# then the same with IME=/usr/bin (vendor package) and IME=llama.cpp-fafix/build-fafix/bin (with the fix)
```

Our results, 7B Q4_K_M, 2 chunks: unpatched token-by-token **0.2148 / 81.96 %**; `-fa off` 0.0289 /
91.57 %; with the fix 0.0290 / 91.77 %; batched ~0.03. If your unpatched token-by-token result is near
0.03, the bug has been fixed in your build.

## 2. Speed

```bash
taskset -c 0-7 llama.cpp/build-cpu/bin/llama-bench -m qwen2.5-7b-instruct-q4_k_m.gguf -p 512 -n 128 -r 3 -t 8
llama.cpp-fafix/build-fafix/bin/llama-bench -m qwen2.5-7b-instruct-q4_k_m.gguf -p 512 -n 128 -r 3 -t 8
/usr/bin/llama-bench -m qwen2.5-7b-instruct-q4_k_m.gguf -p 512 -n 128 -r 3      # vendor; -t is ignored
# context depth: add -d 2048 or -d 8192 (this llama-bench has -d, not -c)
```

Ours at 2.0 / 1.8 GHz: generic CPU 9.07 / 3.10 tok/s (pp512 / tg128), fixed IME2 40.09 / 5.12, vendor
42.4 / 5.06 ([`data/speed_llama_bench.csv`](../data/speed_llama_bench.csv)).

## 3. Accuracy against F16 and the Q4_0 file

```bash
# F16 reference logits on the generic CPU build (15.2 GB through mmap on 8 GB: ~125 s per chunk)
taskset -c 0-7 llama.cpp/build-cpu/bin/llama-perplexity -m qwen2.5-7b-instruct-fp16-00001-of-00004.gguf \
    -f text.txt -c 512 --chunks 20 -t 8 --kl-divergence-base f16.kld
# Q4_0 with the default mix (output tensor Q6_K)
llama.cpp/build-ime/bin/llama-quantize qwen2.5-7b-instruct-fp16-00001-of-00004.gguf qwen2.5-7b-instruct-q4_0.gguf Q4_0 8
# compare, token-by-token, on the fixed IME2 build
llama.cpp-fafix/build-fafix/bin/llama-perplexity -m qwen2.5-7b-instruct-q4_0.gguf -f text.txt -c 512 \
    -b 512 -ub 1 -t 8 --kl-divergence-base f16.kld --kl-divergence
```

Do not run quantization and a 20-chunk F16 comparison at the same time on an 8 GB board (OOM).

## 4. Boost / turbo correctness check

```bash
/usr/bin/llama-perplexity -m qwen2.5-7b-instruct-q4_k_m.gguf -f text.txt -c 512 --chunks 4 --kl-divergence-base vendor4.kld
echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost
for i in 1 2; do /usr/bin/llama-perplexity -m qwen2.5-7b-instruct-q4_k_m.gguf -f text.txt -c 512 --chunks 4 \
    --kl-divergence-base vendor4.kld --kl-divergence | grep -E '^(Mean    KLD|Maximum KLD)'; done
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

At nominal clocks every repeat must give mean KLD 0.000000 against the saved run (ours: 60 of 60 runs).
Our boost evidence came from a slightly different comparison (2 chunks, against upstream CPU token-by-token
logits): two identical boosted runs gave mean KLD 124 and 0.089, the same run at nominal clocks 0.029.

## 5. Embedding server memory growth

```bash
/usr/bin/llama-server -m qwen3-embedding-0.6b-q8_0.gguf --embedding --pooling last -c 4096 -b 4096 -ub 4096 \
    --host 127.0.0.1 --port 8090 &
python3 embed-memory-growth.py --pid $! --text long_text.txt
```

Vendor 0.1.9 on our board: 1,733 MB → 5,381 MB after the first pass, still growing in the second pass;
upstream: flat after the first pass ([`data/embedding_server_memory_growth.csv`](../data/embedding_server_memory_growth.csv)).

## 6. IME2 kernel stand

See [`ime2-kernel-check/README.md`](ime2-kernel-check/README.md): it checks the IME2 matmul kernels of a
built `libggml-cpu.so` on real Q4_K / Q6_K / Q4_0 weights.
