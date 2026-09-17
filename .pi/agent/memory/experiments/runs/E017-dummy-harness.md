# E017 - shape-faithful qwen4exp dummy harness runs on gfx1201

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- tool: `.pi/agent/memory/experiments/tools/mkq4expdummy.py` (new; experiment-local, not upstream code)
- artifact: `models/q4exp-4l.gguf`, 12.76 GB, 113 tensors (ignored by `.gitignore:69` `/models/*`,
  so it can never be staged by accident; `models/` otherwise holds tracked vocab fixtures)
- generate: `--layers 4 --ple-head-rows 3145728` (~1 min, all types written directly)
- ref data: `../results/user/gguf-dump.log` (real file header dump from the bench, 1224 tensors)

## what it is

A GGUF that reproduces the real model's **names, shapes and per-tensor types**, with zero-filled
data, reduced to the first 4 layers. Not the in-tree fixture: `test-llama-archs` emits a
metadata-only model that the loader materialises as f32 with random data
(`src/llama-model-loader.cpp:1292-1325`), which cannot express real bytes-per-token or a row-cut
table. Geometry comes from `src/models/qwen4exp.cpp`; types come from the real file's dump.

Real per-token gather bytes are preserved (16 rows x 110 B), `head_dim=256` with `n_embd=2560`
and 24 heads needs explicit `attention.key_length`/`value_length` (2560/24 = 106 otherwise), and
`ssm.inner_size` must equal `value_dim` = 6144 or the GDN assert at `:878` fires.

## result

| build | table (rows/head) | file | tg128 @ d40960 | pp4096 @ d40960 |
|---|---|---|---|---|
| 4 layers, real table geometry | 35.2 GB (20,000,096) | 42.4 GB | 182.14 +/- 0.64 | 6017.49 +/- 34.35 |
| 4 layers, row-cut | 3.7 GB (2,097,152) | 10.9 GB | 182.71 +/- 1.14 | **6406.93 +/- 85.02** |
| 4 layers, chosen build | 5.5 GB (3,145,728) | 12.8 GB | **182.40 +/- 0.55** | not measured |
| smoke, 32 experts | 0.1 GB | 1.6 GB | 232.59 (not comparable) | 12934.29 |

Decode is **flat across a 10x table range** (182.14 / 182.71 / 182.40 - noise) while prefill is
~6.5% faster with the small table. That is the first direct measurement of the n-gram table's
paging effect, and it matches E011's bound from the other direction: nothing per token, cost
visible when thousands of rows stream at once. The 3.7 GB figure was cut too aggressively for a
realistic scatter distance, so the harness now carries 5.5 GB.

## what the harness can and cannot be used for

Per-step cost here is 1000/182.71 = **5.47 ms**. At 640 GB/s the GPU math for one token is about
0.63 ms (lm_head, 635 M params) + ~0.19 ms (4 layers x 10 experts of Q4_K/Q5_0) ~ 0.8 ms, so
roughly **4.5 ms/step is per-step host work even in this minimal 4-layer graph** - the same class
E011 isolated at ~28 ms on the bench. It does not reproduce the *magnitude*: per-node host work
scales with layer count (48 vs 4 layers), so the dummy is a delta instrument for GPU-side changes
and a graph/placement A/B instrument, **not** a stand-in for the bench box's absolute numbers.

Also unmeasurable here: cross-GPU collectives and split modes (`llm_arch_supports_sm_tensor`
is false for qwen4exp on this tree, `src/llama-arch.cpp:1130`), and the recurrent state is
fixed-size so state bandwidth never binds.

Cosmetic: `general.file_type` is unset so llama-bench labels it "all F32" and "?B (guessed)".
