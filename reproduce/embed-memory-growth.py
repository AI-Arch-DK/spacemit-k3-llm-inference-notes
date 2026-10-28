#!/usr/bin/env python3
"""Watch the memory of a `llama-server --embedding` process over requests of different lengths.

Run on the board, next to the server. Two passes over the same request lengths (16, 32, ... tokens; the
second pass uses different text), then short queries. One JSON line per request with the server's
VmRSS / VmHWM. Stops when RSS exceeds --stop-mb, so the board does not run out of memory.

On a SpacemiT K3 (8 GB), llama.cpp-tools-spacemit 0.1.9 with Qwen3-Embedding-0.6B Q8_0 and -c 4096 grew
from 1.7 GB to 5.4 GB in the first pass and kept growing in the second one; upstream llama.cpp stayed flat.

Usage:
  llama-server -m qwen3-embedding-0.6b-q8_0.gguf --embedding --pooling last -c 4096 -b 4096 -ub 4096 --port 8090 &
  python3 embed-memory-growth.py --pid $! --text some_long_text.txt [--url http://127.0.0.1:8090]
The text file must hold at least 2 x (sum of lengths) tokens of any prose (~120 KB for the defaults).
Requests are token slices of that text turned back into text with /detokenize, so the server may count a
few tokens more or less than n.
"""
import argparse
import json
import sys
import time
import urllib.request

LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4000]


def post(url, path, payload, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def mem(pid):
    out = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "RssAnon"):
                out[k] = int(v.split()[0]) // 1024  # kB -> MB
    return {"rss_mb": out.get("VmRSS"), "hwm_mb": out.get("VmHWM"), "anon_mb": out.get("RssAnon")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--pid", type=int, required=True, help="PID of the llama-server process")
    ap.add_argument("--text", required=True, help="plain-text file to cut requests from")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--short", type=int, default=30, help="number of short queries at the end")
    ap.add_argument("--stop-mb", type=int, default=6500)
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    url = a.url.rstrip("/")
    proxy_free = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # ignore HTTP_PROXY
    urllib.request.install_opener(proxy_free)

    _, tok = post(url, "/tokenize", {"content": open(a.text, encoding="utf-8").read()})
    tokens = tok["tokens"]
    need = 2 * sum(lengths) + 16 * a.short
    if len(tokens) < need:
        sys.exit(f"text too short: {len(tokens)} tokens, need {need}")

    def emit(rec):
        rec.update(mem(a.pid))
        print(json.dumps(rec), flush=True)
        return rec["rss_mb"] > a.stop_mb

    if emit({"step": "start"}):
        return
    off = 0
    steps = [("len", p, n) for p in (1, 2) for n in lengths] + [("short", 3, 16)] * a.short
    for step, pss, n in steps:
        _, d = post(url, "/detokenize", {"tokens": tokens[off:off + n]})
        off += n
        t0 = time.time()
        code, _ = post(url, "/v1/embeddings", {"input": d["content"]})
        if emit({"step": step, "pass": pss, "n": n, "code": code, "sec": round(time.time() - t0, 2)}):
            print(json.dumps({"step": "stop", "why": f"rss over {a.stop_mb} MB"}), flush=True)
            return


if __name__ == "__main__":
    main()
