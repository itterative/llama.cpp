#!/usr/bin/env python3
"""E061 screen: per-instantiation VGPR / occupancy / spill report for mmvf.cu.

Compiles one TU the way the build does and prints the compiler's kernel-resource-usage
remark per instantiation, filtered to the shapes we care about. The point of the screen is
that VGPRs must go *up* with MMVF_K_UNROLL (evidence that the extra loads are really live)
while spills stay 0 and occupancy stays at the wave limit.

  python3 resource-screen.py <unroll> [<unroll> ...]
"""
import re, shlex, subprocess, sys, os

BUILD = "/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp/build"
CLANG = "/usr/lib64/rocm/llvm/bin/clang++"
FLAGS = f"{BUILD}/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/flags.make"

def flags():
    txt = open(FLAGS).read()
    def get(var):
        m = re.search(rf"^{var} = (.*)$", txt, re.M)
        return shlex.split(m.group(1).replace('"', ""))
    return get("HIP_DEFINES") + get("HIP_INCLUDES") + get("HIP_FLAGS")

def report(unroll):
    args = flags() + [f"-DMMVF_K_UNROLL={unroll}", "-x", "hip",
                      "-Rpass-analysis=kernel-resource-usage", "-c",
                      f"{BUILD}/../ggml/src/ggml-cuda/mmvf.cu", "-o", "/dev/null"]
    p = subprocess.run([CLANG] + args, capture_output=True, text=True, cwd=BUILD)
    if p.returncode != 0:
        print(f"unroll={unroll}: BUILD FAILED\n" + "\n".join(p.stderr.splitlines()[-15:]))
        return
    cur, out = None, []
    for line in p.stderr.splitlines():
        m = re.search(r"Function Name: (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        v = re.search(r"VGPRs: (\d+)", line)
        if v and cur:
            out.append((cur, int(v.group(1))))
    # attach the rest of the metrics per function
    metrics = {}
    cur = None
    for line in p.stderr.splitlines():
        m = re.search(r"Function Name: (\S+)", line)
        if m:
            cur = m.group(1); metrics.setdefault(cur, {})
            continue
        for key in ("VGPRs", "TotalSGPRs", "VGPRs Spill", "SGPRs Spill", "Occupancy \\[waves/SIMD\\]", "LDS Size \\[bytes/block\\]"):
            r = re.search(rf"{key}: (\d+)", line)
            if r and cur:
                metrics[cur][key.replace("\\", "").split(":")[0]] = int(r.group(1))
    print(f"\n=== MMVF_K_UNROLL={unroll} ===")
    print(f"{'instantiation':60s} {'VGPR':>5s} {'spill':>6s} {'occ':>4s} {'LDS':>5s}")
    for name in sorted(metrics):
        # only the decode shapes with the block sizes we benchmark
        if ", Li1E" not in name and "Li1E" not in name:
            continue
        if "Li256E" not in name and "Li32E" not in name:
            continue
        d = metrics[name]
        short = re.sub(r"^_ZL13mul_mat_vec_f", "mmvf", name)[:58]
        print(f"{short:60s} {d.get('VGPRs','?'):>5} {d.get('VGPRs Spill','?'):>6} "
              f"{d.get('Occupancy [waves/SIMD]','?'):>4} {d.get('LDS Size [bytes/block]','?'):>5}")

for u in sys.argv[1:] or ["1"]:
    report(u)