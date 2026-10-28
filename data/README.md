# Data

Summary tables built from the raw measurement files (llama-perplexity and llama-bench reports, server
logs, monitor logs). Raw logs and reference logit files (74 MB – 1.5 GB each) are not included.

| file | what |
|---|---|
| [`kld_results.csv`](kld_results.csv) | every KL-divergence run: build, mode (batched / token-by-token), reference, clocks, chunks, mean / median / 99 % / max KLD, same top-1 %, PPL ratio. Rebuilt directly from the `llama-perplexity` reports. The last row summarises 60 repeated runs (one result signature) |
| [`speed_llama_bench.csv`](speed_llama_bench.csv) | `llama-bench` pp512 / tg128 for every build, model, quantization and clock setting, with date and system state |
| [`ime2_matmul_error.tsv`](ime2_matmul_error.tsv) | IME2 kernel stand output: relative L2 error per tensor, weight type and activation kind (see `reproduce/ime2-kernel-check/README.md` for the columns) |
| [`ime2_check_tensors.tsv`](ime2_check_tensors.tsv) | which tensors (first 256 rows) the stand used |
| [`embedding_server_memory_growth.csv`](embedding_server_memory_growth.csv) | RSS / HWM / anon memory of the vendor and upstream embedding servers after each request of the controlled test (two passes of 16…4,000 tokens, then short queries) |
| [`embedding_long_texts.csv`](embedding_long_texts.csv) | synthetic long texts (Russian prose `ru_*`, Python code `code_*`): token counts, cosine vendor↔upstream, vendor↔Ollama (`num_ctx` 8192), Ollama default↔8192 (truncation at 4,095 tokens), seconds per text |
| [`firmware_opp_tables.txt`](firmware_opp_tables.txt) | X100/A100 OPP tables (MHz/mV) from the factory and updated device trees, `turbo-mode` OPPs |
| [`f16_reference_sha256.txt`](f16_reference_sha256.txt) | sha256 and sizes of the F16 reference GGUF split files |

Units: KLD in nats; speeds in tokens/s; memory in MB (MiB from `/proc`); clocks as X100 / A100 GHz.
"upstream" = llama.cpp `0a1ef30`; "vendor" = `llama.cpp-tools-spacemit` 0.1.9 unless the row says otherwise.
