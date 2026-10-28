# Patches

## 0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch

Fixes wrong token-by-token decode (duplicated tokens, KLD to F16 0.286 instead of ~0.05) on the SpacemiT
IME2 backend of llama.cpp by initialising every element of the running maximum `M[QLEN]` in
`flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow` (`ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:669`).
Details and measurements: [docs/decode-bug-flash-attention-ime2.md](../docs/decode-bug-flash-attention-ime2.md).

```bash
cd llama.cpp
git apply /path/to/this/repo/patches/0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch
# or: patch -p1 < .../0001-ggml-cpu-spacemit-init-all-running-max-in-fa-mrow.patch
```

Checked to apply cleanly to llama.cpp `0a1ef30` (b11796) and master `f5222f2` (2026-10-28) [V]. The build
we measured on the board carried the same two-line change (with a different comment). The vendor package
`llama.cpp-tools-spacemit` is built from the same backend; rebuilding it with the patch was not tried.

License: MIT ([LICENSE-CODE](../LICENSE-CODE)), the same as llama.cpp.
