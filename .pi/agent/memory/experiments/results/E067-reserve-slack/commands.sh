#!/bin/bash
# E067 - reserve slack sweep. VRAM sampled from sysfs during each run, golden PPL per arm.
set -u
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib
T=$PWD/.pi/agent/memory/experiments/results/E067-reserve-slack
S=${E067_SAMPLER:-/tmp/pi-coder-scratchpad-obx3lc/e067}
mkdir -p $S
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
VRAM=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
B="build/bin/llama-cli -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024 -n 1"

rm -f $T/*.log
for arm in c0 c50 c100; do
    case $arm in c0) S="";; c50) S="GGML_GALLOCR_RESERVE_SLACK=50";; c100) S="GGML_GALLOCR_RESERVE_SLACK=100";; esac
    for i in 1 2 3; do
        timeout 600 sh -c "while :; do cat $VRAM 2>/dev/null; sleep 0.05; done" > $S/vram-$arm-$i.txt 2>/dev/null &
        sampler=$!
        env $S GGML_PROF_REGIONS=1 timeout 900 $B -f $F < /dev/null > $T/$arm-$i.log 2>&1
        kill $sampler 2>/dev/null
        GGML_PROF_REGIONS=1 timeout 900 build/bin/llama-perplexity -m models/q4exp-4l.gguf -f \
            .pi/agent/memory/experiments/tools/golden-corpus.md > $T/$arm-$i-ppl.log 2>&1
    done
done
echo done
