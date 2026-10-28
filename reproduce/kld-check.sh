#!/bin/bash
# Show the SpacemiT IME2 decode defect with llama-perplexity KL divergence.
# Reference: the generic CPU build on X100, token-by-token. Compared: an IME2 build token-by-token (as is
# and with -fa off) and batched.
# Usage: MODEL=qwen2.5-7b-instruct-q4_K_M.gguf TEXT=text.txt CPU=llama.cpp/build-cpu/bin \
#        IME=llama.cpp/build-ime/bin [CHUNKS=2] [OUT=kld-out] bash kld-check.sh
#   IME=/usr/bin uses the vendor package llama.cpp-tools-spacemit; IME=llama.cpp-fafix/build-fafix/bin the fix.
# Expected without the fix: token-by-token Mean KLD ~0.21, Same top p ~82 %; with -fa off or the fix ~0.03 / ~92 %.
set -euo pipefail
MODEL=${MODEL:?path to the GGUF}
TEXT=${TEXT:?path to a plain-text file, ~100 KB}
CPU=${CPU:?bin dir of the generic CPU build}
IME=${IME:?bin dir of the IME2 build}
CHUNKS=${CHUNKS:-2}
OUT=${OUT:-kld-out}
mkdir -p "$OUT"
REF=$OUT/ref_ub1_c${CHUNKS}.kld

if [ ! -f "$REF" ]; then
    taskset -c 0-7 "$CPU/llama-perplexity" -m "$MODEL" -f "$TEXT" -c 512 --chunks "$CHUNKS" -b 512 -ub 1 -t 8 \
        --kl-divergence-base "$REF" > "$OUT/ref.out" 2> "$OUT/ref.err"
fi

run() {
    local tag=$1; shift
    "$@" -m "$MODEL" -f "$TEXT" -c 512 --chunks "$CHUNKS" -t 8 --kl-divergence-base "$REF" --kl-divergence \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err"
    printf '%-18s %-28s %s\n' "$tag" "$(grep -m1 '^Mean    KLD' "$OUT/$tag.out" || echo 'no KLD line, see .err')" \
        "$(grep -m1 '^Same top p' "$OUT/$tag.out" || true)"
}

run token-by-token "$IME/llama-perplexity" -b 512 -ub 1
run token-fa-off   "$IME/llama-perplexity" -b 512 -ub 1 -fa off
run batched        "$IME/llama-perplexity"
