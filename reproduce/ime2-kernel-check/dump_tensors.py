#!/usr/bin/env python3
"""Dump the first R weight rows of a layer from a GGUF (as stored, quantised) and the layer's RMSNorm weights.

Input for the ime2_check stand: real Q4_K / Q6_K blocks of qwen2.5-7b-instruct-q4_K_M.gguf, so the
distribution of sub-block scales and minimums (which the vendor backend's repack depends on) comes from the
model, not from a random generator.

Usage: GGUF_PY=<llama.cpp>/gguf-py MODEL=<model.gguf> python3 dump_tensors.py [out_dir]
Output: <out>/<name>.bin (raw row bytes) and <out>/manifest.tsv: name, type, ne0, rows, bytes per row.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py"))  # llama.cpp/gguf-py
from gguf import GGUFReader  # noqa: E402

MODEL = os.environ.get("MODEL", "qwen2.5-7b-instruct-q4_K_M.gguf")
LAYERS = (0, 14, 27)
WEIGHTS = ("attn_q", "ffn_gate", "ffn_down", "attn_v", "attn_output", "ffn_up")
NORMS = ("attn_norm", "ffn_norm")
ROWS = 256


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tensors")
    out.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(MODEL)
    by_name = {t.name: t for t in r.tensors}
    lines = ["name\ttype\tne0\trows\trow_bytes"]
    for layer in LAYERS:
        for w in WEIGHTS + NORMS:
            name = f"blk.{layer}.{w}.weight"
            t = by_name[name]
            ne0 = int(t.shape[0])
            data = t.data  # quantised: (rows, bytes per row); F32: (ne0,)
            if t.tensor_type.name == "F32":
                raw = data.tobytes()
                rows, row_bytes = 1, len(raw)
            else:
                rows = min(ROWS, data.shape[0])
                raw = data[:rows].tobytes()
                row_bytes = data.shape[1]
            (out / f"{name}.bin").write_bytes(raw)
            lines.append(f"{name}\t{t.tensor_type.name}\t{ne0}\t{rows}\t{row_bytes}")
    (out / "manifest.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
