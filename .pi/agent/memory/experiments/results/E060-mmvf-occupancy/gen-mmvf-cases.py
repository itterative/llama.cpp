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

# (k, m, type, what it is in the decode window, us/call from the trace)
SHAPES = [
    (2560,   512, "f32",  "ffn_gate_inp router (mirrored)",      27.4),
    (2560,   512, "bf16", "indexer.q_proj (mirrored)",           23.7),
    (10240,  320, "bf16", "output_hc_down (mirrored) control",   15.2),
    (320,  10240, "bf16", "output_hc_up (mirrored) control",     22.1),
    (10240,    4, "bf16", "hc_*_inject (mirrored)",               5.8),
    (2560,   128, "bf16", "indexer.k_proj (mirrored)",            2.5),
    (2560,    12, "f32",  "ssm_alpha/beta card slice",            2.5),
    (2560,     1, "f32",  "unidentified f32 row",                 2.2),
]

def case(k, m, tname, label):
    ts = TSIZE[tname]
    enum = ENUM[tname]
    b0 = ts
    b1 = ts * k
    b2 = b1 * m
    # integer formatting only: the harness reads these with operator>> into an integer,
    # so a scientific-notation field is silently truncated to its mantissa
    src0 = f"{enum} {k} {m} 1 1 {b0:.0f} {b1:.0f} {b2:.0f} {b2:.0f}"
    y0 = TSIZE["f32"]
    src1 = f"0 {k} 1 1 1 {y0:.0f} {y0*k:.0f} {y0*k:.0f} {y0*k:.0f}"
    head = f"29 0 {m} 1 1 1 16 " + "0 "*16 + "2 "
    name = f"mmvf_{tname}_k{k}_m{m}"
    return head + src0 + " " + src1 + f" {name}  # {label}"

def main():
    for k, m, tname, label, _ in SHAPES:
        print(case(k, m, tname, label))
    print(f"# {len(SHAPES)} cases", file=sys.stderr)

main()