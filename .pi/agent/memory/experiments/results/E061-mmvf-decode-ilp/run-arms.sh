#!/bin/bash
# E061: build and screen each MMVF_K_UNROLL arm on the dev box.
# The knob is compile-time (register arrays), so each arm is a rebuild of one TU, as in E059.
# The screen is regression-only: E060 showed the bench's regime (cold weight, few blocks) cannot
# be built on this box, so this checks correctness, the census/probe cases and the dummy tg.
set -u
R=/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
T=$R/.pi/agent/memory/experiments/results/E061-mmvf-decode-ilp
SRC=$R/ggml/src/ggml-cuda/mmvf.cu
CASES=$R/.pi/agent/memory/experiments/results/E060-mmvf-occupancy/mmvf-cases.txt
export LD_LIBRARY_PATH=$R/build/bin
export Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1

mkdir -p $T/arms

for U in 1 2 4; do
    sed -i "s/^#define MMVF_K_UNROLL .*/#define MMVF_K_UNROLL $U/" $SRC
    if ! cmake --build $R/build --target ggml-hip -j 8 > $T/arms/build_u$U.log 2>&1; then
        echo "U=$U BUILD FAILED"; continue
    fi
    md5sum $R/build/bin/libggml-hip.so | awk -v u=$U '{print "u"u, $1}' >> $T/arms/md5.txt

    # op-level: the 23 census + probe cases (noise floor is bs0 vs bs0r in E060)
    GGML_CUDA_MMVF_BLOCK_SIZE=0 timeout 900 $R/build/bin/test-backend-ops perf \
        --test-file $CASES -b ROCm0 -o MUL_MAT > $T/arms/u$U-cases.log 2>&1

    # end to end on the dummy, interleaved later; here one -r 3 reading per arm
    timeout 600 $R/build/bin/llama-bench -m $R/models/q4exp-4l.gguf -ngl 99 -lm none -sm none \
        -fa 1 -lzm on-direct -p 0 -n 128 -r 3 -d 4096 > $T/arms/u$U-dummy.log 2>&1

    timeout 300 $R/build/bin/llama-perplexity -m $R/models/q4exp-4l.gguf \
        -f $R/.pi/agent/memory/experiments/tools/golden-corpus.md > $T/arms/u$U-ppl.log 2>&1

    echo "U=$U done"
done

sed -i "s/^#define MMVF_K_UNROLL .*/#define MMVF_K_UNROLL 1/" $SRC
cmake --build $R/build --target ggml-hip -j 8 > $T/arms/build_restore.log 2>&1
echo "restored"
python3 $R/.pi/agent/memory/experiments/results/E060-mmvf-occupancy/parse-mmvf-perf.py $T/arms/u1-cases.log $T/arms/u2-cases.log $T/arms/u4-cases.log
grep -h "tg128" $T/arms/u*-dummy.log
grep -h "Final estimate" $T/arms/u*-ppl.log