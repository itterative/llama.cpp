#!/bin/bash
# E060: run the mmvf census shapes once per forced block size.
# The knob is host-side, so all arms share one binary and one .so - no rebuilds, no RPATH
# trouble. bs0 is the heuristic control; it is repeated last as bs0r to measure the day's
# noise floor at these shapes.
set -u
R=/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
T=$R/.pi/agent/memory/experiments/results/E060-mmvf-occupancy
export LD_LIBRARY_PATH=$R/build/bin
BIN=$R/build/bin/test-backend-ops
CASES=$T/mmvf-cases.txt

[ -s "$CASES" ] || python3 $T/gen-mmvf-cases.py > $CASES

md5sum $R/build/bin/libggml-hip.so | tee $T/arm-md5.txt

for pair in 0:bs0 32:bs32 64:bs64 96:bs96 128:bs128 160:bs160 0:bs0r; do
    bs=${pair%%:*}
    tag=${pair##*:}
    GGML_CUDA_MMVF_BLOCK_SIZE=$bs timeout 900 $BIN perf \
        --test-file $CASES -b ROCm0 -o MUL_MAT > $T/$tag.log 2>&1
    echo "$tag done (block size $bs), rc=$?"
done

python3 $T/parse-mmvf-perf.py $T/bs*.log