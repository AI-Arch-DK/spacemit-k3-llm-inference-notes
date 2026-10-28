#!/bin/bash
# Run a program as an HMP "AI thread", i.e. on the A100 cores (cpu8-15) of a SpacemiT K3.
# The SpacemiT kernel keeps ordinary threads on X100; writing 0 (the calling thread) or a PID to
# /proc/set_ai_thread moves that thread to the A100 cluster, and children inherit it.
# Do not use cgroups/AllowedCPUs=8-15 for this: it bypasses HMP (kernel WARNING in hmp.c).
# The vendor llama.cpp-tools-spacemit and llama.cpp built with GGML_CPU_RISCV64_SPACEMIT move their own
# worker threads to A100 and do not need this wrapper.
# Usage: bash run-on-a100.sh <program> [args...]
echo 0 > /proc/set_ai_thread || { echo "cannot write /proc/set_ai_thread (not a SpacemiT HMP kernel?)" >&2; exit 1; }
exec "$@"
