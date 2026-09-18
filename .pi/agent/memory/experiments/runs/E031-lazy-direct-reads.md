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

Their current command uses `-lzm auto`, which for the 47.7 GiB table already means lazy mmap reads,
so `auto` vs `on-direct` is the apples-to-apples pair - the only difference is who issues the reads.
Same for `-ot per_layer_token_embd=CPU`, which stays: the gather is host-side either way.

Expect more than the PR's +75% at pp2048 if H11 is right, since their page cache cannot hold 111 GiB
while the PR author's could not hold 27 GiB either, but the PR measured a mostly-warm case. Two
things to watch: the `-v` load line (row size, offset, reader count - which also settles the real
table geometry) and whether the host has enough cores, since a gather now costs 2 x cores threads
plus a memcpy of every duplicate row per ubatch.

## Relation to H12

If this recovers prefill, H12 (splitting the table across the four cards' free VRAM) is no longer
the fix for the stall, only the fix for the streaming itself - a strictly larger but also larger
project. H11's majflt/s sampling still distinguishes them: on-direct should cut majflt/s to nearly
nothing during pp while leaving VRAM placement untouched.
