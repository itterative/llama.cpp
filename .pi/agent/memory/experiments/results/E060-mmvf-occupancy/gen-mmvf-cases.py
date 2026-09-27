#!/usr/bin/env python3
"""E060 arm instrument: MUL_MAT perf cases at the mmvf shapes the decode window shows.

Serialized format (tests/test-backend-ops.cpp, test_generic_op; see the E059 record):
  <op> <dst_type> <ne0..3> <n_params> <params...> <num_src> [<src_type> <ne0..3> <nb0..3>]* <name>
op 29 = MUL_MAT, 0 = f32, 30 = bf16. Grid.x = the output row count, one row per block, so
these cases reproduce both the shape and the launch geometry of the real rows.

Shapes are the per-card slices from byte-census.py. The last two are the control: large m,
where the kernel already reaches 291-427 GB/s and no arms should move it much.
"""
import sys

TSIZE = {"f32": 4.0, "bf16": 2.0}
ENUM  = {"f32": 0, "bf16": 30}

# (k, m, type, batch, what it is in the decode window)
#
# batch > 1 puts B slices on ne2 of both sources, so one node reads B *different* m*k
# weight blocks. That is the only way this harness can read cold: a plain case repeats one
# node thousands of times per timed call, so its weight is L2-resident and the measurement
# oversells (E060: 7.33 us here against 27.37 us in the bench decode window). Per-slice cost
# is us/run / batch. Caveat: a batched node launches one grid of (m, B) blocks, so the B
# slices are concurrent, which is not the bench's single 512-block launch - read the batch
# sweep as the MLP ramp, not as the router's own number.
SHAPES = [
    (2560,   512, "f32",  1, "ffn_gate_inp router (mirrored)"),
    (2560,   512, "bf16", 1, "indexer.q_proj (mirrored)"),
    (10240,  320, "bf16", 1, "output_hc_down (mirrored) control"),
    (320,  10240, "bf16", 1, "output_hc_up (mirrored) control"),
    (10240,    4, "bf16", 1, "hc_*_inject (mirrored)"),
    (2560,   128, "bf16", 1, "indexer.k_proj (mirrored)"),
    (2560,    12, "f32",  1, "ssm_alpha/beta card slice"),
    (2560,     1, "f32",  1, "unidentified f32 row"),
    # L2 sweep: same k, m large enough that the weight passes the 8 MB L2 on its own
    (2560,  1024, "f32",  1, "L2 sweep m=1024 (10.5 MB)"),
    (2560,  2048, "f32",  1, "L2 sweep m=2048 (21 MB)"),
    (2560,  4096, "f32",  1, "L2 sweep m=4096 (42 MB)"),
    (2560,  8192, "f32",  1, "L2 sweep m=8192 (84 MB)"),
    (2560,  5120, "f32",  1, "cache bracket m=5120 (52.4 MB)"),
    (2560,  6144, "f32",  1, "cache bracket m=6144 (62.9 MB)"),
    (2560,  7168, "f32",  1, "cache bracket m=7168 (73.4 MB)"),
    # cold batch: the router shape, read cold via B distinct slices
    (2560,   512, "f32",  4, "router shape, B=4 (21 MB)"),
    (2560,   512, "f32", 16, "router shape, B=16 (84 MB)"),
    (2560,   512, "f32", 64, "router shape, B=64 (335 MB)"),
    (2560,   512, "bf16", 16, "indexer.q shape, B=16 (42 MB)"),
    # concurrency probe: keep m=512 (so the grid stays 512 blocks, one row each) and grow k
    # until the weight passes the 64 MB Infinity Cache on its own. Isolates "512 blocks is
    # not enough in flight" from "the weight was in cache".
    (8192,   512, "f32",  1, "probe k=8192 m=512 (16.8 MB)"),
    (16384,  512, "f32",  1, "probe k=16384 m=512 (33.5 MB)"),
    (32768,  512, "f32",  1, "probe k=32768 m=512 (67 MB, cold, 512 blocks)"),
    (65536,  512, "f32",  1, "probe k=65536 m=512 (134 MB, cold, 512 blocks)"),
]

def case(k, m, tname, batch, label):
    ts = TSIZE[tname]
    enum = ENUM[tname]
    b0 = ts
    b1 = ts * k
    b2 = b1 * m          # one k x m weight slice
    b3 = b2 * batch
    # integer formatting only: the harness reads these with operator>> into an integer,
    # so a scientific-notation field is silently truncated to its mantissa
    src0 = f"{enum} {k} {m} {batch} 1 {b0:.0f} {b1:.0f} {b2:.0f} {b3:.0f}"
    y0 = TSIZE["f32"]
    src1 = f"0 {k} 1 {batch} 1 {y0:.0f} {y0*k:.0f} {y0*k:.0f} {y0*k*batch:.0f}"
    head = f"29 0 {m} 1 {batch} 1 16 " + "0 "*16 + "2 "
    name = f"mmvf_{tname}_k{k}_m{m}_b{batch}"
    return head + src0 + " " + src1 + f" {name}  # {label}"

def main():
    for k, m, tname, batch, label in SHAPES:
        print(case(k, m, tname, batch, label))
    print(f"# {len(SHAPES)} cases", file=sys.stderr)

main()