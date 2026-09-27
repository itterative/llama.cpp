-- E059 trace re-analysis: every number in the record that came off the two rocprofv3
-- decode windows. DuckDB because the kernel name field contains commas (see the E059
-- record and the duckdb memory note).
--
--   duckdb /tmp/e059.db < trace-queries.sql
--
-- The first block loads the two kernel traces (5,568,516 rows each = 4 agents x
-- 1,392,129 dispatches) and the two marker traces. 1.8 GB each, ~5 s to load.

CREATE OR REPLACE TABLE kt_base AS SELECT * FROM read_csv(
  '/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/.pi/agent/memory/experiments/results/user/llama-bench/d133df7d4/traces-baseline/result_mmvq_d133df7d4_baseline_kernel_trace.csv',
  header=true, quote='"');
CREATE OR REPLACE TABLE kt_opt AS SELECT * FROM read_csv(
  '/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/.pi/agent/memory/experiments/results/user/llama-bench/d133df7d4/traces-optimized/result_mmvq_d133df7d4_optimized_kernel_trace.csv',
  header=true, quote='"');
CREATE OR REPLACE TABLE mt AS
  SELECT 'base' arm, Domain, Function, Start_Timestamp st, End_Timestamp en, End_Timestamp - Start_Timestamp dur
    FROM read_csv('/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/.pi/agent/memory/experiments/results/user/llama-bench/d133df7d4/traces-baseline/result_mmvq_d133df7d4_baseline_marker_api_trace.csv', header=true, quote='"')
  UNION ALL
  SELECT 'opt', Domain, Function, Start_Timestamp, End_Timestamp, End_Timestamp - Start_Timestamp
    FROM read_csv('/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/.pi/agent/memory/experiments/results/user/llama-bench/d133df7d4/traces-optimized/result_mmvq_d133df7d4_optimized_marker_api_trace.csv', header=true, quote='"');

CREATE OR REPLACE TABLE k AS
  SELECT 'base' arm, Agent_Id agent, Kernel_Name, Start_Timestamp st, End_Timestamp en,
         End_Timestamp - Start_Timestamp dur, Grid_Size_X gx, Grid_Size_Y gy,
         Workgroup_Size_X wx, Workgroup_Size_Y wy
    FROM kt_base
  UNION ALL
  SELECT 'opt', Agent_Id, Kernel_Name, Start_Timestamp, End_Timestamp,
         End_Timestamp - Start_Timestamp, Grid_Size_X, Grid_Size_Y,
         Workgroup_Size_X, Workgroup_Size_Y FROM kt_opt;

-- one row per step: the 385 phase:decode windows (1 warmup test_gen + 3 x 128 measured)
CREATE OR REPLACE TABLE step AS
  SELECT arm, row_number() OVER (PARTITION BY arm ORDER BY st) i, st, en, dur FROM mt
    WHERE Function = 'phase:decode';

-- 1. per-agent device time. This is the correction: 16.02 -> 14.47 ms per card per
--    step over the four agents (1.55 ms saved), not 15.99 -> 14.64 (Agent 1 only).
SELECT arm, agent, round(sum(dur) / 1e6, 1) ms_total,
       round(sum(dur) / 1e6 / 385, 2) ms_per_step
FROM k GROUP BY arm, agent ORDER BY arm, agent;

-- 2. the domain_stats KERNEL_DISPATCH total IS the sum of the four agents' raw spans -
--    there is no row the per-agent filter drops. Expected: 24663.3 / 22286.4 ms.
SELECT arm, round(sum(dur) / 1e6, 1) ms_all_agents FROM k GROUP BY arm;

-- 3. background term: kernels whose name AND call count are identical in both arms are
--    1.83% faster in the optimized arm (285.0 of the 2376.9 ms saving).
CREATE OR REPLACE TABLE same AS
  SELECT a.Kernel_Name FROM
    (SELECT Kernel_Name, count(*) n FROM kt_base GROUP BY 1) a
    JOIN (SELECT Kernel_Name, count(*) n FROM kt_opt GROUP BY 1) b USING (Kernel_Name)
  WHERE a.n = b.n;
SELECT arm, round(sum(dur) / 1e6, 1) ms_unchanged
FROM k WHERE Kernel_Name IN (SELECT Kernel_Name FROM same) GROUP BY arm ORDER BY arm;

-- 4. marker scoping: 43.4% (base) / 41.4% (opt) of Agent 1 device work executes
--    between markers, so device time must never be scoped to marker ranges.
SELECT arm,
  round(sum(CASE WHEN inside THEN dur ELSE 0 END) / 1e6, 1) ms_inside,
  round(sum(CASE WHEN inside THEN 0 ELSE dur END) / 1e6, 1) ms_between
FROM (SELECT k.*, EXISTS (SELECT 1 FROM step s WHERE s.arm = k.arm AND k.st >= s.st AND k.en <= s.en) inside
      FROM k WHERE agent = 'Agent 1')
GROUP BY arm ORDER BY arm;

-- 5. the long spans (warmup, one-step subgraph-launch stall at steps 3 and 258) and the
--    gaps that are not decode work: the 17 s gap is llama-bench's rep-1 depth prefill
--    with the profiler window shut, the 150 ms ones are the state restore.
SELECT arm, i, round(ms, 1) ms, round(gap_before_ms, 1) gap_before_ms FROM (
  SELECT arm, i, dur / 1e6 ms,
         (st - lag(en) OVER (PARTITION BY arm ORDER BY st)) / 1e6 gap_before_ms
  FROM step) WHERE ms > 200 OR gap_before_ms > 50;

-- 6. H19 in the benchmark harness: a graph re-reserve inside every rep start.
SELECT arm, round(st / 1e9, 3) s, round(dur / 1e6, 2) ms
FROM mt WHERE Function IN ('graph:alloc', 'graph:build') ORDER BY arm, st;

-- 7. the collective is a barrier: nearest-neighbour pairing of the four agents' ar_oneshot
--    ENDS lands within a median of 0.0-0.5 us and p05/p95 of -3..+6 us. Pairing by index
--    does not work, ~6% of collectives slip by one period (~230 us).
CREATE OR REPLACE TABLE ar AS
  SELECT arm, agent, st, en, en - st dur FROM k WHERE Kernel_Name LIKE 'void ggml_cuda_ar_oneshot_kernel%';
SELECT b.arm, b.agent, count(*) n,
       round(quantile_cont(b.d_end, 0.5) / 1000, 2) p50_us,
       round(quantile_cont(b.d_end, 0.05) / 1000, 2) p05_us,
       round(quantile_cont(b.d_end, 0.95) / 1000, 2) p95_us
FROM (
  SELECT x.arm, x.agent, x.en, arg_min(x.en - a.en, abs(x.en - a.en)) d_end
  FROM ar x JOIN ar a ON a.arm = x.arm AND a.agent = 'Agent 1' AND abs(a.en - x.en) < 3e6
  WHERE x.agent <> 'Agent 1' GROUP BY x.arm, x.agent, x.en) b
GROUP BY b.arm, b.agent ORDER BY b.arm, b.agent;

-- 8. the straggler: Agent 4 has the shortest collective total and the smallest worst wait,
--    and with ar_oneshot removed the four agents agree to under 1% on every other kernel.
SELECT arm, agent, round(sum(dur) / 1e6, 1) ms, round(max(dur) / 1000, 1) max_us
FROM ar GROUP BY arm, agent ORDER BY arm, agent;

-- 9. mmvq launch geometry. Grid_Size_X is reported in threads (blocks * 32), so
--    blocks = gx / 32 and nrows = blocks * rows_per_block, with rows_per_block =
--    small_k ? wy : 1 (mmvq-config-rdna4.cuh). Check against a known tensor: the opt
--    row ty14 nar0 blocks=2560 is attn_qkv, m = 10240/4 = 2560, and the row ty6 nar2
--    blocks=5120 with wy=2 is hc_*_up at the full m = 10240, i.e. mirrored.
SELECT arm, regexp_extract(Kernel_Name, '\(ggml_type\)([0-9]+)', 1) ty,
       coalesce(nullif(regexp_extract(Kernel_Name, ', (\d+)>\(', 1), ''), '?') nar,
       gx / 32 blocks, gy, wy, count(*) / 4.0 / 385 per_card_step,
       round(avg(dur) / 1000, 2) us
FROM k WHERE Kernel_Name LIKE 'void mul_mat_vec_q%'
GROUP BY arm, ty, nar, gx, gy, wy ORDER BY arm, per_card_step DESC;

-- 10. mmvf launch geometry, same convention. The router is the f32 row with nrows = 512
--     (the full expert count, so it is mirrored) at 27.6 us and 5.24 MB per call.
SELECT regexp_extract(Kernel_Name, 'mul_mat_vec_f<([^>]*)>', 1) tmpl, gx / wx nrows,
       count(*) / 4.0 / 385 per_card_step, round(avg(dur) / 1000, 2) us
FROM k WHERE arm = 'opt' AND Kernel_Name LIKE '%mul_mat_vec_f<%'
GROUP BY tmpl, gx, wx ORDER BY per_card_step DESC;

-- 11. the shape of decode: 63% of kernels under 2 us holding 18% of device time.
SELECT arm, count(*) n,
       round(quantile_cont(dur, 0.5) / 1000, 2) p50_us, round(quantile_cont(dur, 0.9) / 1000, 2) p90_us,
       round(quantile_cont(dur, 0.999) / 1000, 2) p999_us,
       round(100.0 * sum(CASE WHEN dur < 2000 THEN 1 ELSE 0 END) / count(*), 1) pct_under_2us,
       round(100.0 * sum(CASE WHEN dur < 2000 THEN dur ELSE 0 END) / sum(dur), 1) dev_share_under_2us
FROM k WHERE agent = 'Agent 1' GROUP BY arm ORDER BY arm;