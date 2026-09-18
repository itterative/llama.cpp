# E031 - gather lazy tensor rows with direct reads (upstream PR 29030)

Cherry-picked into this branch as `84b141ac6` (`-x`, author Piotr Wilkin), on top of the base the PR
sits on, so it applied with no conflicts. It replaces H11's mechanism at the source: instead of
`ggml_get_rows` demand-faulting a page per ~110-byte row through the mmap, `--lazy-mode on-direct`
dedupes and sorts an ubatch's row indices and reads them with positional `pread`s on 2 x cores
buffered file handles, dequantizing to F32 host-side.

## Dev box validation (RX 9070, dummy `q4exp-4l.gguf`, 5.28 GiB table)

Engages as intended: `add_lazy_reader: tensor per_layer_token_embd.weight direct reads enabled:
50331648 rows of 110 bytes at offset 904147040 of models/q4exp-4l.gguf, 32 readers`.

| arm | pp512 @ d2048 | pp2048 @ d2048 |
|---|---|---|
| `-lzm on` | 7139.45 +- 210.92 | 8237.70 +- 264.83 |
| `-lzm on-direct` | **8734.65 +- 232.59 (+22.3%)** | **8996.09 +- 65.15 (+9.2%)** |

Both well outside the ~2% local noise floor, and the shape matches the PR's own (larger win at short
prompts). Numerics unchanged: golden PPL `263113.6984 +/- 3043.13362` in both arms, so the staged
rows are bit-identical to what `ggml_get_rows` produced.

Note this moved the golden gate's *conditions*, not its value: `-lm mmap` is now a meaningful axis,
whereas every earlier gate arm used `-lm none`.

## Bench arm (planned)

Their current command uses `-lzm auto`, which for the 32.8 GiB table already means lazy mmap reads,
so `auto` vs `on-direct` is the apples-to-apples pair - the only difference is who issues the reads.
Same for `-ot per_layer_token_embd=CPU`, which stays: the gather is host-side either way.

Expect more than the PR's +75% at pp2048 if H11 is right, since their page cache cannot hold 111 GiB
while the PR author's could not hold 27 GiB either, but the PR measured a mostly-warm case. Two
things to watch: the `-v` load line (row size, offset, reader count - which also settles the real
table geometry) and whether the host has enough cores, since a gather now costs 2 x cores threads
plus a memcpy of every duplicate row per ubatch.

## Bench result - H11 confirmed

`results/user/results-lzm-on-direct.log`, build `84b141ac6` (11071):

```
llama-bench -m .../Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf -lm none -sm tensor -fa 1
  -lzm on-direct -ot per_layer_token_embd=CPU -d 131072 -p 512,4096,8192 -n 128 -r 3 -b 2048 -ub 1024
```

No `Q4EXP_SPARSE_FA`, so the comparator is the 11062 arm run with `Q4EXP_SPARSE_FA=0` (the flag-on
control of the same build agreed to +-0.08%, so the choice does not move anything). 11071 differs
from 11062 only by this pick, so this is a clean single-variable A/B:

| test | 11062 `-lzm auto` | 11071 `-lzm on-direct` | delta |
|---|---|---|---|
| pp512 @ 131k | 476.17 +- 29.78 | **653.18 +- 35.59** | **+37.2%** |
| pp4096 @ 131k | 759.35 +- 14.73 | **830.79 +- 6.44** | **+9.4%** |
| pp8192 @ 131k | 850.87 +- 10.32 | 836.72 +- 0.57 | -1.7% (flat) |
| tg128 @ 131k | 21.31 +- 0.47 | 21.13 +- 0.51 | -0.8% (flat) |

The dev box shows the same shape with `-lm none` as with mmap (`lazy read enabled` and `direct reads
enabled` both print, golden PPL unchanged), so `lm` is not what carries the effect.

Three conclusions:

- Prefill really was fault-bound on the table, which is what H11 said and what E027 could not
  otherwise explain: with the cards 26% busy, no GPU-side saving can show up.
- The shape matches the PR's own (big win at short prompts, shrinking as compute takes over) and
  pp8192 is already past the crossover on this box, so prefill there is compute-limited, not I/O.
- Decode unchanged is a real negative result and worth keeping: a tg step touches 16 rows per PLE
  layer, so faults were never the decode cost. The E011 fixed per-step host cost is elsewhere, which
  keeps the host-side decode work (H9, the qsa fast path's remaining ~2 ms) in the queue rather than
  de-prioritising it behind this.

`-lm none` does not disable lazy reading - the table's ranges get mapped regardless, which is why
these arms hold a 111 GiB model on a 62.7 GiB box at all. The bench box cannot use `-lm mmap` (load
time), and does not need to.

## Relation to H12

If this recovers prefill, H12 (splitting the table across the four cards' free VRAM) is no longer
the fix for the stall, only the fix for the streaming itself - a strictly larger but also larger
project. H11's majflt/s sampling still distinguishes them: on-direct should cut majflt/s to nearly
nothing during pp while leaving VRAM placement untouched.
