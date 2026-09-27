#!/usr/bin/env python3
"""Parse the Phase A perf logs into a type x k x arm table."""
import re, os, glob

T = "/tmp/pi-coder-scratchpad-G2Djjx/tbo/phaseA"
BLCK = {"q4_0":32,"q4_1":32,"q5_0":32,"q5_1":32,"q8_0":32,"iq4_nl":32,
        "q2_K":256,"q4_K":256,"q5_K":256,"q6_K":256,"iq4_xs":256}
ORDER = ["q4_0","q4_1","q5_0","q5_1","q8_0","iq4_nl","q2_K","q4_K","q5_K","q6_K","iq4_xs"]
ARMS = [(8,1),(8,0),(4,1),(4,0),(2,1),(2,0),(1,1)]

ANSI = re.compile(r"\x1b\[[0-9;]*m")

def parse(path):
    txt = ANSI.sub("", open(path, errors="replace").read())
    names = re.findall(r"MUL_MAT_ID\(name=([A-Za-z0-9_]+)", txt)
    times = re.findall(r"([0-9]+\.[0-9]+) us/run", txt)
    return dict(zip(names, [float(t) for t in times]))

data = {}
for nw, sk in ARMS:
    p = f"{T}/nw{nw}_sk{sk}.log"
    data[(nw,sk)] = parse(p) if os.path.exists(p) else {}

rows = []
for t in ORDER:
    for k in ([256,1024,2560] if BLCK[t]==256 else [160,640,2560]):
        name = f"{t}_k{k}"
        vals = [data[a].get(name) for a in ARMS]
        if all(v is None for v in vals):
            continue
        rows.append((t, k, k//BLCK[t], vals))

hdr = f"{'type':7} {'k':>5} {'kblk':>4} " + " ".join(f"{f'nw{n}s{s}':>8}" for n,s in ARMS) + "   best"
print(hdr); print("-"*len(hdr))
for t,k,kb,vals in rows:
    got = [v for v in vals if v is not None]
    best = min(got)
    bi = vals.index(best)
    cells = " ".join((f"{v:8.2f}" if v is not None else f"{'-':>8}") for v in vals)
    star = "*" if bi != 0 else " "
    print(f"{t:7} {k:5} {kb:4} {cells}   nw{ARMS[bi][0]}s{ARMS[bi][1]} {star}")

print()
print("baseline = nw8s1 (current default). '*' = baseline is not the best arm.")
n_base_best = sum(1 for _,_,_,v in rows if v and v[0] is not None and min(x for x in v if x is not None) == v[0])
print(f"baseline best in {n_base_best}/{len(rows)} cells")
for i,(nw,sk) in enumerate(ARMS):
    wins = sum(1 for _,_,_,v in rows if v and v[i] is not None and min(x for x in v if x is not None) == v[i])
    tot  = sum(1 for _,_,_,v in rows if v and v[i] is not None)
    print(f"  nw{nw}s{sk}: best in {wins}/{tot}")
