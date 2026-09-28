#!/bin/bash
# E065 - dev-box pp/tg phase inventory. Run from the repo root.
# pp: 1 token after the prompt, window closes at the first decode batch. tg: same prompt, 256 tokens,
# window opens at the first decode batch (the 8k prompt runs untraced before it).
set -u
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib
T=$PWD/.pi/agent/memory/experiments/results/E065-dev-box-phase-inventory
M=models/q4exp-4l.gguf
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
B="build/bin/llama-cli -m $M -f $F -c 8192 -ngl 99 -st --temp 0"

for i in 1 2 3; do
    GGML_PROF_REGIONS=1 timeout 900 $B -n 1   < /dev/null > $T/pp-untraced-$i.log 2>&1
    GGML_PROF_REGIONS=1 timeout 900 $B -n 256 < /dev/null > $T/tg-untraced-$i.log 2>&1
done

for i in 1 2; do
    GGML_PROF_REGIONS=1 GGML_PROF_WINDOW=pp timeout 1800 rocprofv3 --selected-regions --marker-trace \
        --kernel-trace --stats --output-format csv -o $T/pp-traced-$i -- $B -n 1 < /dev/null > $T/pp-traced-$i.log 2>&1
    GGML_PROF_REGIONS=1 GGML_PROF_WINDOW=tg timeout 1800 rocprofv3 --selected-regions --marker-trace \
        --kernel-trace --stats --output-format csv -o $T/tg-traced-$i -- $B -n 256 < /dev/null > $T/tg-traced-$i.log 2>&1
done
echo done
