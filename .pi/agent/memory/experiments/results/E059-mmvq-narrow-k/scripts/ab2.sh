set -u
R=/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
H=$R/ggml/src/ggml-cuda/mmvq-config-rdna4.cuh
W=/tmp/pi-coder-scratchpad-G2Djjx/ab
cd $R
export LD_LIBRARY_PATH=$R/build/bin Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1
cp $H $W/header.pristine
for cycle in 1 2; do
  for arm in base treat; do
    if [ "$arm" = base ]; then
      sed -i "s/^#define MMVQ_RDNA4_NARROW_K_MAX .*/#define MMVQ_RDNA4_NARROW_K_MAX 0/" $H
      sed -i "s/^#define MMVQ_RDNA4_MID_K_MAX .*/#define MMVQ_RDNA4_MID_K_MAX 0/" $H
    else
      cp $W/header.pristine $H
    fi
    cmake --build build --target ggml-hip -j 8 > $W/b_$arm.log 2>&1
    rc=$?
    h=$(md5sum build/bin/libggml-hip.so.0.24.0 | cut -c1-8)
    out=$(timeout 600 ./build/bin/llama-bench -m models/q4exp-4l.gguf -ngl 99 -p 0 -n 128 -r 3 \
            -fa 1 -lzm on-direct -sm none 2>/dev/null | grep tg128 | sed 's/\xc2\xb1.*//' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "cycle$cycle $arm rc=$rc so=$h tg128=$out"
  done
done
cp $W/header.pristine $H
cmake --build build --target ggml-hip -j 8 > $W/b_final.log 2>&1
echo "restored rc=$?"
