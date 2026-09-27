#!/usr/bin/env python3
"""Parse test-backend-ops perf logs into us/run and real GB/s per case and arm.

The GB/s column in the tool is allocated bytes over time (op_flops is 0 for a file-loaded
case), so recompute from the shape in the case name instead.
"""
import re, sys, glob, os

TSIZE = {"f32": 4.0, "bf16": 2.0}
CASE  = re.compile(r"name=(mmvf_(\w+)_k(\d+)_m(\d+))")
RUN   = re.compile(r"(\d+) runs -\s+([\d.]+) us/run")

def parse(path):
    out, cur = {}, None
    for line in open(path, errors="replace"):
        m = CASE.search(line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {"k": int(m.group(3)), "m": int(m.group(4)), "t": m.group(2)})
        # the name and the timing line are separate lines for most cases, one line for some
        r = RUN.search(line)
        if r and cur:
            out[cur]["runs"] = int(r.group(1))
            out[cur]["us"] = float(r.group(2))
    for v in out.values():
        v["bytes"] = v["k"] * v["m"] * TSIZE[v["t"]]
        v["gbps"] = v["bytes"] / (v["us"] * 1e3) if "us" in v else 0.0
    return out

def main():
    logs = sorted(sys.argv[1:]) or sorted(glob.glob("bs*.log"))
    arms = {}
    for p in logs:
        arms[os.path.basename(p).replace(".log", "")] = parse(p)
    if not arms:
        sys.exit("no logs")
    first = next(iter(arms.values()))
    names = sorted(first, key=lambda n: -first[n]["m"])
    hdr = "case".ljust(30) + "".join(f"{a:>10s}" for a in arms)
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        base = first[n]["us"]
        row = f"{n:30s}"
        for a in arms:
            v = arms[a].get(n, {}).get("us")
            row += f"{v:9.2f} " if v else f"{'-':>10s}"
        print(row + f"  bytes={first[n]['bytes']/1e3:.0f}kB")
    print("\nGB/s of the fastest arm per case:")
    for n in names:
        best = min((arms[a][n]["us"], a) for a in arms if n in arms[a])
        v = arms[best[1]][n]
        base = first[n]["us"]
        print(f"  {n:30s} best {best[1]:8s} {v['us']:8.2f} us  {v['gbps']:7.0f} GB/s"
              f"  ({base/v['us']:.2f}x vs first arm)")

main()