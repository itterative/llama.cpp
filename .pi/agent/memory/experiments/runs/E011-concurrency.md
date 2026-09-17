# E011 - decode has ~30 ms of per-step fixed cost

- date: 2026-09-17
- machine: bench-4x-r9700-32g
- tier: T2
- status: done
- parent: E008
- build: `c9a59ef73` (11009), user's fork, same as E005/E006/E008
- tool: `llama-batched-bench` (not `llama-bench` - that has no concurrency knob; the sweep is
  `-npl`)
- command, verbatim: `-lm none -sm tensor -fa 1 -lzm auto -ot per_layer_token_embd=CPU -c 40960
  -npp 512,4096,8192 -ntg 128 -npl 1,2,4`
- raw: `../results/user/results-batched.log` (untracked `.log`)

## the result that matters

Decode step cost barely grows with how many tokens the step carries. From the pp8192 rows
(`T_TG / 128` = ms per step, independent of B):

| B (parallel seqs) | ms/step | ms/token | marginal per added seq |
|---|---|---|---|
| 1 | 32.4 | 32.4 | - |
| 2 | 36.7 | 18.3 | 4.3 ms |
| 4 | 52.5 | 13.1 | 7.9 ms |

Fitting `cost = F + B*m` on B=1,2 gives **F ~ 28 ms/step**, which then under-predicts B=4
(45.2 vs 52.5) - i.e. the fixed part is real and the per-token part grows from ~4 ms to ~8 ms
as the GPUs finally start to bind. Same shape in all three `npp` groups (F ~ 28-32 ms,
per-token 32.4 -> 17.2 -> 11.8 at pp512; 32.4 -> 18.3 -> 13.1 at pp8192).

**So ~90% of a decode step is cost that does not depend on the number of tokens in it.** That
is the host-bound signature E008/E009/E010 were aimed at, and it is established by timing
alone - no debug hooks, no rebuild. Bandwidth, collectives, split mode and graph capture are
already excluded; this says the remaining time is per-step, not per-token, which further
implies graph build / input construction / sync rather than anything streamed.

## A free bonus: an upper bound on the QSA prize for decode

batched-bench has no depth knob, so its `tg` runs at `npp + 128` depth; llama-bench's E005 ran
`tg` at `-d 40960`. Comparing the single-sequence decode rates:

| tool | depth per seq | S_TG t/s |
|---|---|---|
| E011, npl=1 | ~512-8320 | 30.9 - 31.4 |
| E005 | 40960 | 28.2 |

A ~5x context increase costs ~9-10% of decode throughput. So attention-plus-KV over 40 k is
**~10% of a decode token**, which caps what the sparse-attention port (H4b) could return on tg
at that depth - it cannot reach the other ~90%. This is the first quantitative bound on H4b, and
it is a fairly small one for decode.

## Caveat: do not read absolute pp from this run

batched-bench takes no repetition count (`llama-bench` has `-r`), so each row is a single
timing, and the **first row is warmup-contaminated**: pp512 at B=1 reports 127 t/s while the
same `npp` at B=2 reports 399 t/s - a non-physical ordering that can only come from
first-touch costs (table faults, buffer allocation, capture) landing on row one. The tg
columns are consistent across rows and are what this record concludes from.

pp concurrency scaling is otherwise flat and plausible (pp8192: 506.8 -> 526.2 -> 552.0 t/s for
B=1,2,4), i.e. prefill was already saturated at `-ub 512`.

## Found in the header of this log: `-ot` has never done anything

```
W llama_model_loader: tensor overrides do not apply to lazy-read tensors
```

`-ot per_layer_token_embd=CPU` is **inert** for this tensor, because `-lzm auto` classifies the
~30 GB table as lazy-read, and lazy-read forces the CPU buffer type itself
(`lazy_read::buft()`, `src/llama-model-loader.cpp:1080-1086`). The placement the user wanted is
happening, but through lazy mode, not the override.

This corrects F3, which claimed the override was what put the table on the host. It also means
the two have never been separated, and that a clean test now exists (E014): `-lzm off` stops the
lazy mapping so that `-ot` finally applies, giving a **fully resident** RAM copy with no demand
paging. If tg improves, faults were part of the ~28 ms; if it does not, the table's storage mode
is irrelevant and the fixed cost is elsewhere. Costs ~30 GB of permanent RAM, so system RAM
(E008b) is now a prerequisite.
