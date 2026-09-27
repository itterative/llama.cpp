#!/bin/bash
# Sweep the RDNA4 mmvq block-shape knobs at the real qwen4exp MoE shapes.
# Each point: rewrite the two #define defaults, rebuild the single TU, measure 3 shapes.
set -u
R=/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
H=$R/ggml/src/ggml-cuda/mmvq-config-rdna4.cuh
T=/tmp/pi-coder-scratchpad-G2Djjx/tbo
OUT=$T/sweep.txt
cp $H $T/mmvq-config-rdna4.cuh.pristine
cd $R
export LD_LIBRARY_PATH=$R/build/bin
: > $OUT

measure() { # $1 = file
    timeout 200 $R/build/bin/test-backend-ops perf --test-file $T/$1.txt -b ROCm0 -o MUL_MAT_ID 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "[0-9]+\.[0-9]+ us/run" | grep -oE "^[0-9.]+"
}

for nw in 8 4 2 1; do
  for mult in 1 2 4; do
    if [ "$nw" = 1 ] && [ "$mult" != 1 ]; then continue; fi   # small_k needs nwarps > 1
    sed -i "s/^#define MMVQ_RDNA4_NWARPS .*/#define MMVQ_RDNA4_NWARPS $nw/" $H
    sed -i "s/^#define MMVQ_RDNA4_ROWS_MULT .*/#define MMVQ_RDNA4_ROWS_MULT $mult/" $H
    if ! cmake --build build --target ggml-hip -j 8 > $T/build.log 2>&1; then
        echo "nwarps=$nw mult=$mult BUILD FAILED" | tee -a $OUT; continue
    fi
    rows=$((nw*mult)); [ $nw -eq 1 ] && rows=1
    d160=$(measure down_k160); d640=$(measure down_k640); g160=$(measure gate_m160)
    line="nwarps=$nw mult=$mult rows=$rows  down_k160=${d160}us  down_k640=${d640}us  gate_m160=${g160}us"
    echo "$line" | tee -a $OUT
  done
done

cp $T/mmvq-config-rdna4.cuh.pristine $H
cmake --build build --target ggml-hip -j 8 > $T/build.log 2>&1
echo "restored defaults, rebuild rc=$?"
