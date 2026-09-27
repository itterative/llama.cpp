import re, sys

BASE = "/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/.pi/agent/memory/experiments/results/user/llama-bench/d133df7d4"
TY = {'6': 'Q5_0', '8': 'Q8_0', '12': 'Q4_K', '13': 'Q5_K', '14': 'Q6_K'}
STEPS, DEVS = 385, 4

def rows(path):
    d = {}
    for ln in open(path, errors="replace"):
        sig = re.search(r"void mul_mat_vec_q<[^>]*>", ln)
        if not sig:
            continue
        nums = re.findall(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", ln.rstrip())
        if nums:
            c, ms, av, pc = nums[-1]
            d[sig.group(0)] = (int(c), float(ms), float(av))
    return d

def key(s):
    t = re.search(r"ggml_type\)(\d+)", s).group(1)
    a = re.search(r"ggml_type\)\d+, (\d+), (true|false), (true|false), (true|false)(?:, (\d+))?>", s)
    return (TY.get(t, t), a.group(2), a.group(3), a.group(5) or "-")

b = rows(f"{BASE}/traces-baseline/stats.log")
o = rows(f"{BASE}/traces-optimized/stats.log")

print(f"{'type':6} {'fus':6} {'smallk':7} {'narrow':7} {'calls':>8} {'/card/step':>10} {'base us':>8} {'opt us':>8} {'ratio':>6}")
print("-" * 78)
tb = to = 0.0
for s in sorted(set(list(b) + list(o)), key=lambda s: -b.get(s, (0, 0, 0))[1]):
    ty, fu, sk, nw = key(s)
    bc, bms, bus = b.get(s, (0, 0, 0.0))
    oc, oms, ous = o.get(s, (0, 0, 0.0))
    c = bc or oc
    tb += bms; to += oms
    print(f"{ty:6} {fu:6} {sk:7} {nw:7} {c:8} {c/DEVS/STEPS:10.1f} {bus:8.2f} {ous:8.2f} {(bus/ous if ous else 0):6.2f}")

print(f"\nmul_mat_vec_q total: base {tb:.1f} ms -> opt {to:.1f} ms = {tb/to:.3f}x, saved {tb-to:.1f} ms")
print(f"per card per step:   base {tb/DEVS/STEPS:.3f} ms -> opt {to/DEVS/STEPS:.3f} ms, saved {(tb-to)/DEVS/STEPS:.3f} ms")
print(f"call counts: base {sum(v[0] for v in b.values())}  opt {sum(v[0] for v in o.values())}")

# group by narrow band
for band, pred in [("narrow=1 (kblk<=5)", lambda k: k[3] == "1"),
                   ("narrow=2 (kblk<=20)", lambda k: k[3] == "2"),
                   ("narrow=0 (untouched)", lambda k: k[3] == "-")]:
    ob = sum(v[1] for s, v in b.items() if pred(key(s)))
    oo = sum(v[1] for s, v in o.items() if pred(key(s)))
    cb = sum(v[0] for s, v in b.items() if pred(key(s)))
    co = sum(v[0] for s, v in o.items() if pred(key(s)))
    print(f"{band:22} base {ob:8.1f} ms /{cb:7}   opt {oo:8.1f} ms /{co:7}")
