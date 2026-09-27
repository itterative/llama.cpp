#!/usr/bin/env python3
"""Per-card mmvq weight traffic in one decode step, from the real Q4_K_M tensor census.

Reads the gguf dump (all four shards are in the one log) and applies the split axes that
src/llama-model.cpp::get_tensor_config declares, so the per-card byte count is not a guess:

  axis 0/1  -> the tensor is split across the devices on that axis, per card = total/4
  mirrored  -> no pattern matches, every device reads the whole tensor (the default rule)

MoE expert tensors are scaled by expert_used_count / expert_count, because a decode step
touches 10 of the 512 expert slices. Tensors that are not matvec weights at decode
(embeddings via get_rows, the router through topk_moe, 1D norms, the f32/bf16 group which
is mul_mat_vec_f and measured separately) are excluded.

Usage:
  python3 byte-census.py [gguf-dump.log] [mmvq_ms_per_step]

Default log path is the census kept with the E053/E059 results. Second argument is the
mmvq device time per card per step from the decode window (E059: 4.35 ms), used only for
the achieved-bandwidth line.

Reproduces E059's statement: 1487 MB per card per step, 342 GB/s, 53% of the RDNA4
640 GB/s peak, of which 412 MB is read by every card four times over.
"""
import re, sys, collections

TS = {  # type -> (block_size, bytes_per_block)
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
}

# get_tensor_config order: first match wins. axis 0/1 split, M mirrored.
RULES = [
    (r"blk\.\d*\.attn_q\.weight", 1), (r"blk\.\d*\.attn_(k|v)\.weight", 1),
    (r"blk\.\d*\.attn_qkv\.weight", 1), (r"blk\.\d*\.attn_output\.weight", 0),
    (r"blk\.\d*\.attn_gate\.weight", 1), (r"blk\.\d*\.ssm_out\.weight", 0),
    (r"blk\.\d*\.ssm_alpha\.weight", 1), (r"blk\.\d*\.ssm_beta\.weight", 1),
    (r"blk\.\d*\.ssm_conv1d\.weight", 1),
    (r"blk\.\d*\.ffn_up(_exps)?\.weight", 1), (r"blk\.\d*\.ffn_gate(_exps)?\.weight", 1),
    (r"blk\.\d*\.ffn_gate_up(_exps)?\.weight", 1), (r"blk\.\d*\.ffn_down(_exps)?\.weight", 0),
    (r"blk\.\d*\.ffn_up_shexp\.weight", 1), (r"blk\.\d*\.ffn_gate_shexp\.weight", 1),
    (r"blk\.\d*\.ffn_down_shexp\.weight", 0), (r"output\.weight", 1),
]
# not mmvq weights: embeddings are gathered, the router runs in topk_moe, norms and biases
# are 1D, and _inject / indexer.* / hc_*_inject are bf16 (mul_mat_vec_f, not this table)
SKIP = ("norm", "bias", "ssm_a", "ssm_dt", "gate_inp", "per_layer_token_embd",
        "token_embd", "attn_sinks", "inject")
EXPS = re.compile(r"_exps")
TENSOR_LINE = re.compile(
    r"^\s*\d+:\s+\d+:\s+[\d,\s]+\|.*"  # matched per line below, kept simple on purpose
)

def axis_of(name):
    for rx, ax in RULES:
        if re.fullmatch(rx, name):
            return ax
    return "M"

def main():
    log = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/.pi/agent/memory/experiments/results/user/gguf-dump.log"
    ms = float(sys.argv[2]) if len(sys.argv) > 2 else 4.35

    rows = []
    for line in open(log, errors="replace"):
        # <idx>: <size>: <bytes> | <shape> | <TYPE> | <name>
        parts = line.split("|")
        if len(parts) < 4:
            continue
        head, shape, typ, name = (p.strip() for p in parts[:4])
        typ = typ.split()[0] if typ else ""
        if typ not in TS or any(s in name for s in SKIP):
            continue
        if not re.match(r"^\d+:", head):
            continue
        dims = [int(x) for x in shape.split(",") if x.strip()]
        while len(dims) > 2 and dims[-1] == 1:
            dims.pop()
        if len(dims) < 2 or min(dims) < 2:
            continue
        bs, bpb = TS[typ]
        n = 1
        for d in dims:
            n *= d
        total = n / bs * bpb
        ax = axis_of(name)
        per_card = total if ax == "M" else total / 4.0
        act = 10.0 / 512.0 if EXPS.search(name) else 1.0
        rows.append((name, typ, ax, per_card * act))

    agg = collections.defaultdict(lambda: [0.0, 0])
    for name, typ, ax, b in rows:
        key = (re.sub(r"^blk\.\d+\.", "blk.N.", name), typ, ax)
        agg[key][0] += b
        agg[key][1] += 1

    tot = sum(r[3] for r in rows)
    split = sum(r[3] for r in rows if r[2] != "M")
    print(f"{'tensor':34s} {'type':5s} {'axis':4s} {'tensors':>7s} {'MB/card/step':>13s}")
    for (pat, typ, ax), (b, cnt) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"{pat:34s} {typ:5s} {str(ax):4s} {cnt:7d} {b/1e6:13.1f}")
    print(f"\ntotal per card per step   {tot/1e6:8.0f} MB")
    print(f"  split across devices    {split/1e6:8.0f} MB")
    print(f"  mirrored, read 4x       {(tot-split)/1e6:8.0f} MB ({100*(tot-split)/tot:.0f}% of the traffic)")
    print(f"\nover {ms:.2f} ms of mmvq per card per step: {tot/1e6/ms:.0f} GB/s "
          f"= {100*tot/1e6/ms/640:.0f}% of the RDNA4 640 GB/s peak")
    print(f"if the mirrored part were split (same time): {split/1e6/ms:.0f} GB/s on {split/1e6:.0f} MB")

main()