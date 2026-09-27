#!/bin/bash
# Phase A: nwarps x small_k sweep over all 11 RDNA4-whitelisted types at 3 k values each.
set -u
R=/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
H=$R/ggml/src/ggml-cuda/mmvq-config-rdna4.cuh
T=/tmp/pi-coder-scratchpad-G2Djjx/tbo
cd $R
export LD_LIBRARY_PATH=$R/build/bin
mkdir -p $T/phaseA
cp $H $T/mmvq-config-rdna4.cuh.pristine

for nw in 8 4 2 1; do
  sed -i "s/^#define MMVQ_RDNA4_NWARPS .*/#define MMVQ_RDNA4_NWARPS $nw/" $H
  if ! cmake --build build --target ggml-hip -j 8 > $T/phaseA/build_$nw.log 2>&1; then
      echo "nwarps=$nw BUILD FAILED"; continue
  fi
  for sk in 1 0; do
    if [ "$nw" = 1 ] && [ "$sk" = 0 ]; then continue; fi   # small_k needs nwarps > 1, so identical
    GGML_CUDA_MMVQ_RDNA4_SMALL_K=$sk timeout 900 $R/build/bin/test-backend-ops perf \
        --test-file $T/sweep-cases.txt -b ROCm0 -o MUL_MAT_ID > $T/phaseA/nw${nw}_sk${sk}.log 2>&1
    echo "done nwarps=$nw small_k=$sk"
  done
done

cp $T/mmvq-config-rdna4.cuh.pristine $H
cmake --build build --target ggml-hip -j 8 > $T/phaseA/build_restore.log 2>&1
echo "restored, rc=$?"
