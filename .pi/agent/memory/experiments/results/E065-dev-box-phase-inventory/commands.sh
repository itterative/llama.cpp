#!/bin/bash
# E065 - dev-box pp/tg phase inventory. Run from the repo root.
# Harness note: -f does NOT deliver a prompt to llama-cli (params.prompt_file is read by imatrix and
# cvector-generator only); -p "$(cat ...)" does. -c 32768 = the whole sparse corpus (32.68k tokens),
# -c 8192 = the same corpus truncated to the context.
# rocprofv3 CSV output goes to a scratch dir, not the repo: it is large and reproducible from here.
set -u
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib
T=$PWD/.pi/agent/memory/experiments/results/E065-dev-box-phase-inventory
S=${E065_TRACES:-/tmp/pi-coder-scratchpad-obx3lc/e065/traces}
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
P="$(cat $F)"
B="build/bin/llama-cli -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -ub 1024"
mkdir -p $S
rm -f $T/*.log

for c in 32768 8192; do
    for i in 1 2 3; do
        GGML_PROF_REGIONS=1 timeout 900 $B -c $c -n 1   -p "$P" < /dev/null > $T/pp-untraced-c$c-$i.log 2>&1
        GGML_PROF_REGIONS=1 timeout 900 $B -c $c -n 256 -p "$P" < /dev/null > $T/tg-untraced-c$c-$i.log 2>&1
    done
done

for i in 1 2; do
    GGML_PROF_REGIONS=1 GGML_PROF_WINDOW=pp timeout 1800 rocprofv3 --selected-regions --marker-trace \
        --kernel-trace --stats --output-format csv -o $S/pp-traced-$i -- $B -c 32768 -n 1 -p "$P" \
        < /dev/null > $T/pp-traced-$i.log 2>&1
    GGML_PROF_REGIONS=1 GGML_PROF_WINDOW=tg timeout 1800 rocprofv3 --selected-regions --marker-trace \
        --kernel-trace --stats --output-format csv -o $S/tg-traced-$i -- $B -c 32768 -n 256 -p "$P" \
        < /dev/null > $T/tg-traced-$i.log 2>&1
done
echo "done, traces in $S"

# shallow arms, added after the first pass: -c 8192 ABORTED ("request (32673 tokens) exceeds the
# available context size (8192 tokens)") because llama-cli errors instead of truncating. Fixed by
# feeding a 45k-byte prefix of the corpus into -c 16384.
PS="$(head -c 45000 $F)"
for i in 1 2 3; do
    GGML_PROF_REGIONS=1 timeout 900 $B -c 16384 -n 1   -p "$PS" < /dev/null > $T/pp-untraced-c16384-$i.log 2>&1
    GGML_PROF_REGIONS=1 timeout 900 $B -c 16384 -n 256 -p "$PS" < /dev/null > $T/tg-untraced-c16384-$i.log 2>&1
done
