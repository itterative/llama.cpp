#!/bin/bash
# E066 - per-ubatch graph topology probe (LLAMA_UBATCH_DEBUG, added next to process_ubatch).
set -u
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib
T=$PWD/.pi/agent/memory/experiments/results/E066-node-count-probe
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
# -v is required: the probe is a LLAMA_LOG_INFO line and this build defaults to WARN (same trap as the
# pool mode line in experiment-protocol)
B="build/bin/llama-cli -v -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768"

rm -f $T/*.log
LLAMA_UBATCH_DEBUG=1 GGML_PROF_REGIONS=1 timeout 900 $B -ub 1024 -n 1   -f $F < /dev/null > $T/pp-ub1024.log 2>&1
LLAMA_UBATCH_DEBUG=1 GGML_PROF_REGIONS=1 timeout 900 $B -ub 128  -n 1   -f $F < /dev/null > $T/pp-ub128.log  2>&1
LLAMA_UBATCH_DEBUG=1 GGML_PROF_REGIONS=1 timeout 900 $B -ub 1024 -n 256 -f $F < /dev/null > $T/tg-ub1024.log 2>&1
echo done
