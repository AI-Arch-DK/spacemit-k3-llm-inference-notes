# Is the SpacemiT K3 a good embedding server for RAG? (Qwen3-Embedding on IME2, memory growth, long texts)

**Short answer.** Yes for short queries, with care for long ones. The vendor `llama-server --embedding`
(`llama.cpp-tools-spacemit` 0.1.9, A100 / IME2) with Qwen3-Embedding-0.6B Q8_0 gives vectors that match
upstream llama.cpp and Ollama to cosine ≥ 0.9985, answers a single query in **~35 ms over 1GbE**, and is
3.4–4.7× faster than upstream llama.cpp on the X100 cores for short texts. But its memory is **not reused between
requests of different lengths**: a server with `-c 4096` grew to **7.3 GB** while indexing texts of at
most 2,185 tokens, and with `-c 8192` it was **killed by the OOM killer**. Under short queries only, it
plateaus (+53 MB over 8 hours).

Labels: **[V]** measured, **[I]** inference, **[NV]** not verified. Board: K3 Pico-ITX 8 GB, Bianbu 4.0.1,
X100 2.0 GHz / A100 1.8 GHz, boost off, 2026-10-26/27, unless stated otherwise.

- [Quality: vendor IME2 vs upstream vs Ollama](#quality-vendor-ime2-vs-upstream-vs-ollama)
- [Latency and throughput](#latency-and-throughput)
- [Long texts (up to 8k tokens)](#long-texts-up-to-8k-tokens)
- [Memory: the embedding server grows until OOM](#memory-the-embedding-server-grows-until-oom)
- [8-hour service run](#8-hour-service-run)
- [Running an embedder and a generator on one 8 GB board](#running-an-embedder-and-a-generator-on-one-8-gb-board)
- [Bigger embedder: Qwen3-Embedding-4B](#bigger-embedder-qwen3-embedding-4b)
- [Practical settings](#practical-settings)

Server command used throughout (vendor or upstream binary):

```bash
llama-server -m qwen3-embedding-0.6b-q8_0.gguf --embedding --pooling last -c 4096 -b 4096 -ub 4096 \
    --host 127.0.0.1 --port 8090
# OpenAI-style endpoint: POST /v1/embeddings {"input": "..."}
```

## Quality: vendor IME2 vs upstream vs Ollama

Qwen3-Embedding-0.6B Q8_0 (sha256 `06507c7b…`, the same file Ollama ships as `qwen3-embedding:0.6b`),
`--pooling last`, 167 texts (139 paragraphs of technical prose and 28 search queries), 20,005 tokens;
reference: upstream llama.cpp `0a1ef30` on X100, batched [V]:

| set | cosine min / median | top-10 retrieval overlap | top-1 identical | texts/s |
|---|---|---|---|---|
| upstream, one text per request (control) | 0.99909 / 0.99979 | 98.2 % | 28 of 28 | 0.67 |
| vendor, batched | 0.99891 / 0.99967 | 97.9 % | 28 of 28 | 2.26 |
| vendor, one text per request | 0.99924 / 0.99967 | 97.1 % | 28 of 28 | 2.57 |

The vendor path differs from upstream no more than upstream differs from itself between batch and single
requests. Embedding is a single forward pass, so the [decode bug](decode-bug-flash-attention-ime2.md)
does not apply.

**Serving an existing index built elsewhere.** 200 stored vectors from a live index that had been built
with Ollama on x86-64 (same GGUF), recomputed on the board: cosine stored↔board min 0.99882, 1st
percentile 0.99917, median 0.99971 (stored↔fresh Ollama: 1.00000) [V]. The board can serve such an
index **without re-indexing**.

**End to end.** A semantic search over two collections of a SQLite knowledge base (the whole index holds
~1,300 vectors; numpy cosine,
30 queries against each base) on a client that called the board's embedder over the LAN instead of a
local Ollama with the same GGUF: top-1 identical in 30 of 30 for both bases, the same top-3 set in 28 of
30, top-5 overlap 99 %, |Δ top-1 score| ≤ 0.006; query vectors client-Ollama vs board — cosine min 0.99930, median 0.99974 [V].

## Latency and throughput

| measurement | upstream llama.cpp, X100 ×8 | **vendor, A100 (IME2)** | upstream llama.cpp, A100 as AI threads |
|---|---|---|---|
| one short text, local, median of 10 (2026-10-21, 2.2/1.8 GHz) | 240 ms (230–262) | **51 ms** (51–83) | 4,310 ms |
| batch of 32 short texts (2026-10-21) | 15.0 s = 2.1 texts/s | **4.4 s = 7.2 texts/s** | 285 s = 0.1 texts/s |
| one query from a LAN client, 1GbE, incl. a small HTTP shim; n = 150 (2026-10-26) | — | **min 34 / median 35 / p90 63 / max 93 ms** | — |
| 8-hour service, one query every 180 s; n = 133 (2026-10-27) | — | median 61 / p90 72 / max 84 ms | — |

The generic ggml RVV path on the A100 cores is unusable (18× slower than X100): only the vendor IME2
path makes the A100 useful. The model must stay resident: loading and runtime initialisation (TCM,
hugepages) take seconds per process start.

How the LAN numbers were taken: the board was connected to the client through a USB Ethernet adapter
(1GbE, 941 Mbit/s both ways measured with iperf3); a small Python shim on the board translated the
client's Ollama-style `/api/embeddings` calls to `/v1/embeddings`. Measurements were taken only in
windows where the client's CPU was below 15 % busy.

## Long texts (up to 8k tokens)

Synthetic Russian prose and Python code cut to 257…8,001 tokens, plus long records of a real index;
vendor server `-c 4096` (and a fresh `-c 8192` process per text above 4,096 tokens), upstream `-c 8192`,
Ollama with default settings and with `num_ctx 8192` ([`data/embedding_long_texts.csv`](../data/embedding_long_texts.csv)) [V]:

- **Vendor vs upstream: cosine 0.9996–0.99985 at every length**, including 6,145 and 7,013 tokens: no
  outliers on long batches for the embedder.
- Vendor is faster at every length, but the lead shrinks on long texts: 1,025 tokens — 2.6–2.8 s vs 14.7–14.8 s
  (5.2–5.6×); 2,049 — 7.7–8.1 s vs 34.8 s (4.3–4.5×); 4,097 — 43–45 s vs 91 s (2.0–2.1×); 6,145 — 91.9 vs
  169.8 s (1.85×); 7,013 — 115.7 vs 208.3 s (1.8×).
- Vendor vs Ollama without truncation: 0.9985–0.99987.
- 8,001 tokens were not sent to the vendor server (does not fit in 8 GB, see below).
- A 4,097-token text on a `-c 4096` server is rejected with `exceeds the available context size`: the
  window has to be larger than the text plus one special token.

**Ollama silently truncates embedding input at 4,095 tokens with default settings** (`num_ctx` 4096;
`prompt_eval_count` = 4095). Truncated vs full vector: 0.9990–0.9997 at 4,097 tokens, 0.996–0.998 at
4,340–4,863, 0.963–0.980 at 6,145, 0.965 at 7,013, 0.923–0.946 at 8,001 [V, Ollama on x86-64]. If you
compare a K3 embedder with Ollama on long texts, set `num_ctx` first.

## Memory: the embedding server grows until OOM

RSS of the server process, Qwen3-Embedding-0.6B Q8_0 [V]:

| server | RSS after start | after the longest text | over a series of requests |
|---|---|---|---|
| vendor `-c 2048` | 1,503 MB | 2,806 MB (2,032 tokens) | — |
| vendor `-c 4096` | 1,725 MB | 4,349 MB (4,080 tokens) | **VmHWM 7,336 MB** after ~150 index records of at most 2,185 tokens |
| vendor `-c 8192` | — | VmHWM 7,489 MB on texts of at most 2,575 tokens | **killed by the OOM killer** (anon-rss 7,621,720 kB) on the next text after a 4,340-token one |
| upstream `0a1ef30`, X100, `-c 8192` | — | VmHWM **2,596 MB** for all 52 long texts up to 8,001 tokens | — |

**Controlled test** (fresh server `-c 4096`; requests of 16, 32, 64 … 3,072, 4,000 tokens, then the same
lengths again with different text, then 30 short queries; stop at RSS > 6.5 GB;
[`data/embedding_server_memory_growth.csv`](../data/embedding_server_memory_growth.csv)) [V]:

| | vendor (IME2, A100) | upstream (CPU, X100) |
|---|---|---|
| after start | 1,733 MB | 1,572 MB |
| after the first pass, 16 → 4,000 tokens | **5,381 MB** (+0.8–1.0 MB per token on long requests) | 1,831 MB |
| second pass, the same lengths | **keeps growing**: 5,818 MB already at 16 tokens, 6,559 MB at 3,072 — stopped by the guard | **+0 MB** |
| 30 short queries | not reached | +0 MB |

The vendor build does not reuse memory even for a repeated request length; upstream allocates buffers
for the largest request once and reuses them. With short queries only (10–20 tokens) the vendor server
reaches a plateau ([below](#8-hour-service-run)). [I] This looks like a backend memory pool
(`mem_backend: HPAGE` in the init line) that never returns memory and fragments: each new long request
takes a new block, short ones fit into blocks already taken. We did not look inside the backend, and we
did not check an upstream build with the IME2 backend in embedding mode. A fresh process for a single
request needs about 1.26 GB + 0.76 MB per token [I, fit over the measurements].

## 8-hour service run

Vendor server, Qwen3-Embedding-0.6B, `-c 4096`; a LAN client sent one query plus two search calls every
180 s, as a RAG front end would; board monitor once a minute [V]:

| | value |
|---|---|
| duration | 8 h continuous (restarted from zero after each of two power interruptions of the test bench) |
| RSS | **1,724 → 1,777 MB**, flat at 1,777 MB after ~3 h |
| memory available | ≥ 6,236 MB throughout |
| temperatures / clocks | 42–45 °C; 2.0 / 1.8 GHz the whole time |
| load average (1 min) | ≤ 2.9 |
| errors | 0 of 160 requests; no new errors in the server log |
| latency | median 61 / p90 72 / max 84 ms (133 requests) |

## Running an embedder and a generator on one 8 GB board

- **7B generation next to the embedder: no.** In our run the board hung: Qwen2.5-7B Q4_K_M generation
  (upstream llama.cpp on X100) was started while the vendor embedder, after 278 requests, had already
  grown. Ping kept answering (0.4–0.7 ms), ssh timed out during banner exchange, the journal of that boot
  shows 12 minutes of silence and **no OOM-kill line** — thrashing without swap, where the kernel evicts
  file pages (program code, mapped weights) and never reaches the OOM killer [I]. Recovery needed a power
  cycle. This was our test design error: that step had no memory guard. 7B on the IME2 path alone already
  reaches max RSS 7.5 GB (file pages included; `free` showed 5.67 GB used, 2.26 GB available) [V].
  **Plan for one model at a time on the 8 GB board.**
- **1.5B generation next to the embedder fits but is slowed down.** Vendor embedder 0.6B (`-c 2048`) under
  a stream of paragraphs plus `llama-bench` upstream Qwen2.5-1.5B Q4_K_M on X100 (`-t 8`), memory guard
  at < 1 GB available (minimum seen: 1,449 MB) [V]:

| | alone | together |
|---|---|---|
| embedder: median latency / throughput | 534 ms / 1.55 texts/s | 537 ms / 1.57 texts/s |
| generator tg128 | 11.70 tok/s | **2.72 tok/s (−77 %)** |
| generator pp512 | 43.6 tok/s | 34.9 tok/s (−20 %) |

Cause [V]: the vendor server process gets the default CPU mask 0–7 (on this kernel normal processes live
on X100 and only "AI threads" move to A100), and under the request stream its **main thread sat on an X100
core at ~35 % CPU** (`exclude_main_thread: 0` in the backend init line). A generator with `-t 8` on the 8
X100 cores waits for the delayed thread at every synchronisation point. Leaving one core free (`-t 7`)
should help [I, not measured]. What `exclude_main_thread` does is an open question to SpacemiT.

## Bigger embedder: Qwen3-Embedding-4B

Q4_K_M (sha256 `2b0cf8f1…`, the file Ollama ships), `-c 2048`, the same 167 texts, batches of 16;
reference upstream on X100 [V]:

| | vendor, A100 | upstream, X100 |
|---|---|---|
| one query from the LAN client, median of 10 | **120 ms** | 723 ms |
| corpus throughput | 0.48 texts/s | 0.13 texts/s |
| VmHWM after the corpus | **6,582 MB** | 4,665 MB |
| cosine vendor vs upstream: min / 1 % / median | **0.958 / 0.967 / 0.980** | — |
| retrieval: top-10 overlap / top-1 | 86.8 % / 28 of 28 | — |

1 − cosine is 40–60 times larger than for the 0.6B Q8_0 model: the Q4_K repack loss
([quantization page](q4k-repack-and-choosing-a-quant.md)) shows directly in the vectors. 4B in Q4_K_M on
IME2 is not recommended; Q8_0 or Q4_0 would avoid the repack loss, but a 4B Q8_0 is ~4.3 GB of weights
[I, not measured].

## Practical settings

- Model: **Qwen3-Embedding-0.6B Q8_0** on the vendor `llama-server --embedding --pooling last`, resident.
- `-c` larger than your longest input + 1; for an index of records up to ~2,600 tokens, 4096 is enough.
- For bulk re-indexing of long texts, either restart the server after each batch, restart it on an RSS
  threshold, or use an upstream `llama-server` on X100 for that job (flat memory, 2–3.5× slower) [I].
- Do not run a 7B generator on the same 8 GB board.
- Python clients using `requests` honour `HTTP_PROXY`/`HTTPS_PROXY`: with a proxy in the environment, the
  first call to the board's LAN address went to the proxy and got **502**. Add the board's address to
  `NO_PROXY` [V].
