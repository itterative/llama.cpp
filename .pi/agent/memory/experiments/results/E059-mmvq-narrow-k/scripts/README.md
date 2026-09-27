# Repro scripts for E059 and E060

Three small tools. Together they make the numbers in `runs/E059-mmvq-narrow-k-rdna4.md`
reproducible instead of hand-derived, and they carry no state of their own.

## `byte-census.py` - per-card weight traffic in one decode step

```sh
python3 byte-census.py                     # all four gguf shards in one log
python3 byte-census.py path/to/gguf-dump.log 4.35
```

Reads the tensor census kept at `results/user/gguf-dump.log` and applies the split axes that
`src/llama-model.cpp::get_tensor_config` declares, so "per card" is a code fact and not a
guess: an unmatched tensor name falls through to `MIRRORED`, which means every device reads
the whole tensor. MoE expert tensors are scaled by 10/512 active experts.

Reproduces, for the bench box Q4_K_M at d16384:

- 1487 MB per card per step of quantized matvec traffic, of which 412 MB (28%) is mirrored
  and read by all four cards;
- 342 GB/s = 53% of the RDNA4 640 GB/s peak over E059's 4.35 ms of `mul_mat_vec_q` per
  card per step (E053 had this at ~115 GB/s, 18%).

## `trace-queries.sql` - everything the record took from the two rocprofv3 windows

```sh
duckdb /tmp/e059.db < trace-queries.sql      # ~5 s to load 2 x 1.8 GB, then the queries
```

Eleven queries, each with the record's figure in the comment above it: per-agent device
time (the 16.02 -> 14.47 correction), the aggregate-equals-per-agent-sum check, the 1.83%
background term, the inside/between-marker split, the step timeline with the 17 s gap and
the rep-start re-reserves, the `ar_oneshot` end alignment and the Agent 4 straggler, and
the mmvq / mmvf launch geometry tables.

## E060 instruments (one directory up, in `results/E060-mmvf-occupancy/`)

`gen-mmvf-cases.py` writes the eight per-card MUL_MAT shapes; `sweep-mmvf-block.sh` runs
them once per forced block size via `GGML_CUDA_MMVF_BLOCK_SIZE` and calls
`parse-mmvf-perf.py`, which recomputes GB/s from the shape because the tool's own column is
allocated bytes over time.

```sh
bash ../../E060-mmvf-occupancy/sweep-mmvf-block.sh
```