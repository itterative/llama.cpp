# Bench-finish bundle - closed out, all four items accounted for

Written 2026-09-27 after E062, closed out the same day. Nothing here needs a run any more: item 1 was
already answered by the untraced `-r 10` logs (`u1/u2/u4-noprof.log` in `results/user/h23-runs-b673ab4ae`),
item 2 arrived as the user's own test run (`results/user/bench-finish-2026-09-27/`, recorded as E063),
item 3 was closed by deployment knowledge and item 4 by an unrecorded post-fix measurement plus the fact
that the recorded loss was itself the pre-E056 reservation tax. The one question this bundle *created* is
E064, which is flag-only and not part of it.

| # | item | record | decides | status |
|---|---|---|---|---|
| 1 | E062 paired `MMVF_K_UNROLL` 4 vs 1 | E062 | keep or revert the branch's first ~2% default flip | **closed without a new run: the untraced `-r 10` arms ARE the confirmation (+1.9-2.0%), flip kept in `8372ffc1f`** |
| 2 | post-flip REF baseline | E026 style, new raw dir | the comparability anchor for every later A/B | **done - E063**: the user's own `-r 10` run, tg 43.43 / 43.15 / 41.88 / 37.64 at d4096/16384/40960/131072, pp8192 2219.27 / 2120.65 / 1958.24 / 1521.66, raw in `results/user/bench-finish-2026-09-27/`. Caveat: `GGML_CUDA_AR_ONESHOT_PROBE=50` and tg-after-pp, which is 6.8% above the same day's E062 U=4 arm - see E063 and E064 |
| 3 | E048 completion: `--spec-draft-n-max` 1/2/3/6 + no-spec | E048 | whether over-drafting is still costing a third of decode | **closed 2026-09-27 without a run:** mtp works, acceptance 0.2-0.9 by task, `n_max = 3` is what is served, so the `n_draft = 6` tax is on nobody's bill |
| 4 | E013 revival: pool on/off vs depth | E013 | whether the now-default pool still loses below ~32k | **closed 2026-09-27 without a run:** the user reports the post-E050-fix behaviour as a slight improvement rather than a loss; the -5.1% at d4096 was the pre-E056 per-ubatch re-reservation |

Prerequisites, all on the bench box:

- tree at `134941f31` or later (`git log -1 --oneline`); the E062 default flip is `8372ffc1f`.
- the usual build path: `bash ~/llamacpp-install.sh rocm`, `llama-bench` on `PATH`.
- model file (the path E061's script used):
  `$HOME/.cache/huggingface/hub/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf`
- save raw logs under `results/user/bench-finish-<date>/`, one file per invocation, unedited.
- paste the logs back so the records get written; do not hand-transcribe numbers.

Hard rules for this bundle, from `../PROTOCOL.md` and E061/E062:

- **No tracer.** `rocprofv3` inverted the sign of the E061 comparison by perturbing the allreduce's
  wait tail. Zero `rocprofv3` invocations here.
- **`GGML_CUDA_AR_ONESHOT_PROBE` unset, explicitly.** E061's arms all set it and E062's untraced logs
  do not echo the env, so it is the one uncontrolled variable left. Use `env -u`.
- **The arms alternate in one session.** Sequential `-r 3` runs in separate invocations cannot resolve
  a 2% effect. Item 1's logs satisfied the `-r 10` half of this (`tok:decode` = 1280 = 10x128) and not the
  recorded-alternation half; E062 accepts that, given both treatment arms agree.
- **Keep the environment explicit even where the code now defaults it on.** The reference environment
  below names `Q4EXP_POOLED`, `GGML_CUDA_MMVQ_RDNA4_SMALL_K` and `LLAMA_LAZY_PREFETCH` at their
  default-on values so a later default flip cannot silently change what "REF" meant.

## Reference environment (REF)

```sh
export M="$HOME/.cache/huggingface/hub/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf"
export REF="GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_BF16=nccl Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1 Q4EXP_POOLED=1 GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 LLAMA_LAZY_PREFETCH=1"
export BENCH="llama-bench -m $M -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -b 2048 -ub 1024 -o md"
```

`Q4EXP_SPARSE_FA` and `GGML_FATTN_RDNA_RTILE` are still default-off; without them the run is not the
configuration the user actually serves with.

## 1. E062 paired confirmation (rebuild per arm)

**Closed without a new run.** The recipe below was written for the bill it would have cost; it is kept
because a re-run should use it, and because the md5 check it names is the one step where both arms can
silently run the same binary. Evidence that closed it: `results/user/h23-runs-b673ab4ae/u[124]-noprof.log`.

`MMVF_K_UNROLL` is a **compile-time macro** in `ggml/src/ggml-cuda/mmvf.cu` (it is `#ifndef`, so no env
var can override it), which is why the arms need a one-TU rebuild each, exactly as in E061. The arm
binary is `libggml-hip.so`; record its md5 per invocation.

Script, run from the repo root:

```sh
set -e
run() {
    local U=$1 PASS=$2
    git checkout -- ggml/src/ggml-cuda/mmvf.cu
    sed -i "s/#    define MMVF_K_UNROLL 4/#    define MMVF_K_UNROLL $U/" ggml/src/ggml-cuda/mmvf.cu
    grep -n "define MMVF_K_UNROLL" ggml/src/ggml-cuda/mmvf.cu          # prove the arm
    bash ~/llamacpp-install.sh rocm >/dev/null 2>&1
    local LIB=$(ldd "$(command -v llama-bench)" | awk '/libggml-hip\.so/{print $3}')
    md5sum "$LIB" | tee -a arms_u$U.md5
    env -u GGML_CUDA_AR_ONESHOT_PROBE GGML_PROF_REGIONS=1 $REF \
        $BENCH -p 0 -n 128 -d 16384 -r 10 \
        | tee "mmvf_u$U-pass$PASS.log"
}
run 4 1; run 1 1; run 4 2; run 1 2
git checkout -- ggml/src/ggml-cuda/mmvf.cu
```

Two checks before trusting anything: the `grep` must print the arm's value, and the two `arms_u*.md5`
files must differ (if not, the rebuild did not pick up the edit and both arms ran the same code).

Extraction per log:

- `tg128 @ d16384` and its spread from the bench table.
- exit region report, `phase:sync` tg bucket (`tg N calls ... ms/call`) - this is the host drain that
  E062 quoted as 16.96 ms/token in the u1 arm, and `phase:decode` ms/call is the host launch side.
- `meta:allreduce` tg calls and ms/call.

Decision rule (already in E062): paired delta of u4 against u1. **Revert** the default if it is worse
than -1%; **keep** it at +1% or better; between -1 and +1% is inconclusive and gets written down as
that, not as a win.

## 2. Post-flip REF baseline

One invocation, no rebuild, no tracer, `GGML_PROF_REGIONS` off for the cleanest t/s:

```sh
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF $BENCH -p 8192 -n 128 -d 4096,16384,40960,131072 -r 10 \
    | tee baseline-$(date +%Y%m%d).log
llama-bench --version 2>&1 | tee -a baseline-$(date +%Y%m%d).log   # or capture the "build:" line
```

This is the number future A/Bs are read against. Record the bin/so md5 in the same file. If time is
short, `-p 8192 -r 10` at d131072 alone is still worth having.

## 3. E048 completion: draft count sweep with acceptance

**Closed 2026-09-27 without a run.** The user reports mtp working with acceptance from 0.2 to 0.9 by task
and serves `n_max = 3` (the `common/common.h` default), so over-drafting is not on any real run's bill -
the question was a property of `n_draft = 6`. Keep the recipe below for the day acceptance has to be
measured rather than trusted, and keep the load-log cache count as a free check.

E048's gap was "no raw output and no acceptance rate reported". `llama-bench` has no speculative
decode, so this is the server, and `--spec-draft-n-max` is a launch flag, so it is **one server launch
per arm**. The server returns the counters in the final response, so record JSON, not log lines.

Arms: no-spec, 1, 2, 3, 6. Use the user's normal server flags; the spec part is
`--spec-type draft-mtp --spec-draft-n-max $N`. Keep `-c` and the flags fixed across arms. Local test
scaffolding is `--spec-default` for ngram types; it must not be set here.

Per arm:

```sh
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF \
llama-server -m $M -lm none -ngl 999 -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU \
    -c 32768 --spec-type draft-mtp --spec-draft-n-max $N &
# keep the user's normal server flags too; only the spec pair changes between arms
# wait for the health endpoint to answer, then 3 requests, nothing else running
for i in 1 2 3; do
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"<one fixed prompt, the same string in all arms>"}],
         "n_predict":512,"temperature":0,"seed":42,"cache_prompt":false}' \
    | tail -1 | tee -a draft_n$N.jsonl | jq -c '.timings'
done
kill %1; wait
```

Record from `.timings`: `prompt_n`, `predicted_n`, `predicted_per_second`, `draft_n`,
`draft_n_accepted`. Per-arm acceptance is `draft_n_accepted / draft_n`; mean accepted length is
`1 + draft_n_accepted / (predicted_n - draft_n_accepted)`. The server's own
`draft acceptance = ... mean len = ...` line is a cross-check when the log is captured.

Read the four arms against each other, not against E046's or E048's numbers - both were taken on
unknown builds and configurations. If the winner is n=1 or 2, that is a real finding: it contradicts
the `n_max = 3` default in `common/common.h`, and the fix is a config default, not code.

Two things the sweep does not answer, note them in the record: draft cost grows with depth because a
draft replay scans the shared KV densely (see H18), so one arm at the user's real `-c 245760` with the
winning `n` is worth a follow-up; and the draft ctx is invisible to `common_perf_print`, so this run
does not split target vs draft time.

## 4. E013 revival: pool on/off versus depth (E050's crossover)

**Closed 2026-09-27 without a run.** The user reports the post-E050-fix behaviour as a slight improvement
rather than a loss, and the recorded -5.1% at d4096 came from the build *before* E056's reservation fix -
whose per-ubatch re-reservation was exactly a short-context tax. Nearest recorded support: E044's dev-box
depth table (d4096 within noise, d16384 +5.6%) and E056's pooled prefill +2.4% over pool-off. The post-fix
decode sweep the user refers to is not in the records; if its raw output exists it should be pasted into
E050 rather than re-run.

E050 measured the pool **negative below ~32k** (-5.1% at d4096) on the pre-reservation-fix build.
Since then the reservation was fixed (E056), prefill was un-gated (E056) and the pool became the
default (`9111adf2c`), so every short-context run now pays whatever that crossover is. Flag-only:

```sh
for pass in 1 2; do
  for pool in 1 0; do
    env -u GGML_CUDA_AR_ONESHOT_PROBE GGML_PROF_REGIONS=1 Q4EXP_POOLED=$pool $REF \
        $BENCH -p 4096 -n 128 -d 512,4096,16384,40960,131072 -r 10 \
        | tee "pool${pool}-pass${pass}.log"
  done
done
```

If d4096/d16384 still favour `Q4EXP_POOLED=0` by more than the 5% bar, the branch needs a documented
rule (depth gate in `qsa_pool_get`, or "set Q4EXP_POOLED=0 below 32k" in the launch notes) rather than
a silent tax. If the gap is inside the bar, E050's row gets a correction and this thread closes.

Also read `sched:realloc_size` counts in both arms: E056 claimed 0 re-reserves when pooled, and
`graph:alloc` tg calls in the E062 logs (19 calls at ~20 ms in a 128-token run) show the H19 ratchet is
still live with the pool on. If pool-off shows more, that is H19's scope, not the pool's.