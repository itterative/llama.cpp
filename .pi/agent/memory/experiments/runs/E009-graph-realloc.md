# E009 - no unexpected graph reallocation (negative)

- date: 2026-09-17 | machine: bench-4x-r9700-32g | tier: T2 | status: done | parent: E011
- build `c9a59ef73`, E005 config, env `GGML_SCHED_DEBUG_REALLOC=1`
- result reported by the user: **run completed with no errors and no output**

This one is a genuine negative rather than an inconclusive silence. The realloc hook calls
`GGML_ABORT` (`ggml/src/ggml-backend.cpp:1611-1622`), so had it fired the run would have
crashed loudly. Clean finish means: no graph reallocation with unchanged size, no
`backend_ids_changed` surprise. So "the scheduler is reallocating a same-size graph every step"
is ruled out.

Worth stating the limit of the instrument, because it is narrower than it sounds: the check sits
inside the failure branch of `ggml_gallocr_alloc_graph`, so it detects *realloc attempts that
fail because size was unchanged*. A fresh allocation, a legitimately different size, or host
work outside the allocator would all pass silently. E010 is what actually bounded the graph
machinery cost.

No raw file - there was no output to keep.
