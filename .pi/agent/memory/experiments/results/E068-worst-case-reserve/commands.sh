#!/bin/bash
# E068 - worst-case reserve (LLAMA_RESERVE_WORST_CASE). VRAM sampled from sysfs, golden PPL per arm.
set -u
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib
T=$PWD/.pi/agent/memory/experiments/results/E068-worst-case-reserve
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
G=.pi/agent/memory/experiments/tools/golden-corpus.md
VRAM=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
B="build/bin/llama-cli -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024"

rm -f $T/*.log $T/vram-*.txt
run() { # arm n_tokens reps env
    local arm=$1 n=$2 reps=$3 env=$4
    for i in $(seq 1 $reps); do
        # sampler dumps go to a scratch dir, never into results/ (root .gitignore keeps *.log out of git, but
# these were small enough to slip through and were committed once - see the E067 incident)
S=${E068_VRAM:-/tmp/pi-coder-scratchpad-obx3lc/e068}
mkdir -p $S
        timeout 600 sh -c "while :; do cat $VRAM 2>/dev/null; sleep 0.05; done" > $S/vram-$arm-$i.txt 2>/dev/null &
        local sampler=$!
        env $env GGML_PROF_REGIONS=1 timeout 900 $B -n $n -f $F < /dev/null > $T/$arm-$i.log 2>&1
        kill $sampler 2>/dev/null
        env $env GGML_PROF_REGIONS=1 timeout 900 build/bin/llama-perplexity -m models/q4exp-4l.gguf -f $G \
            > $T/$arm-$i-ppl.log 2>&1
    done
}
run control   1   3 ""
run wc-pp     1   3 "LLAMA_RESERVE_WORST_CASE=1"
run control-tg 256 2 ""
run wc-tg     256 2 "LLAMA_RESERVE_WORST_CASE=1"
echo done
