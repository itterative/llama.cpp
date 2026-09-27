#!/usr/bin/env python3
"""Generate MUL_MAT_ID perf cases for the mmvq RDNA4 block-shape sweep.

Serialized format (tests/test-backend-ops.cpp:11465, test_generic_op):
  <op> <type> <ne0..3> <n_params> <params...> <num_src> [<src_type> <ne0..3> <nb0..3>]* <name>
op 30 = MUL_MAT_ID, 0 = f32, 26 = i32.
nb[0] = type_size, nb[1] = type_size*(ne0/blck_size), and so on.
"""
import sys

# (enum value, type name, blck_size, type_size). type_size from the static_asserts in
# ggml-common.h; q5_0=22 and q4_K=144 are confirmed against a real exported graph.
TYPES = {
    "q4_0":   (2,  32,  18),
    "q4_1":   (3,  32,  20),
    "q5_0":   (6,  32,  22),
    "q5_1":   (7,  32,  24),
    "q8_0":   (8,  32,  34),
    "q2_K":   (10, 256, 84),
    "q4_K":   (12, 256, 144),
    "q5_K":   (13, 256, 176),
    "q6_K":   (14, 256, 210),
    "iq4_nl": (20, 32,  18),
    "iq4_xs": (23, 256, 136),
}

M          = 2560   # n_embd, the down-projection output rows
N_USED     = 10     # num_experts_per_tok
N_MATS     = 64     # experts held; only n_used are read, this just bounds allocation
N_TOKENS   = 1      # decode

def case(tname, k):
    enum, blck, tsize = TYPES[tname]
    assert k % blck == 0, f"k={k} not representable for {tname} (blck {blck})"
    b0 = tsize
    b1 = tsize * (k // blck)
    b2 = b1 * M
    b3 = b2 * N_MATS
    src0 = f"{enum} {k} {M} {N_MATS} 1 {b0} {b1} {b2} {b3}"
    src1 = f"0 {k} {N_USED} {N_TOKENS} 1 4 {4*k} {4*k*N_USED} {4*k*N_USED}"
    ids  = f"26 {N_USED} {N_TOKENS} 1 1 4 2048 2048 2048"
    head = f"30 0 {M} {N_USED} {N_TOKENS} 1 16 " + "0 "*16 + "3 "
    return head + src0 + " " + src1 + " " + ids + f" {tname}_k{k}"

def main():
    out = []
    for tname, (enum, blck, tsize) in TYPES.items():
        # K-quants need k a multiple of 256; the block-32 types also take the real 4-card k=160.
        ks = [256, 1024, 2560] if blck == 256 else [160, 640, 2560]
        for k in ks:
            out.append(case(tname, k))
    print("\n".join(out))
    print(f"# {len(out)} cases", file=sys.stderr)

main()
