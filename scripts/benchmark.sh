#!/bin/bash

# build if not already
# cargo build --release

# get number of CPU cores (Linux/macOS)
NUM_CORES=$(nproc)

# for macOS, you might use:
# NUM_CORES=$(sysctl -n hw.ncpu)

# for Windows (WSL/Git Bash), nproc might work. otherwise, you'd set it manually:
# NUM_CORES=8 # e.g., for 8 threads

echo ""
echo "--- WARNING: BENCHMARKING MAY TAKE HOURS DEPENDING ON ACCURACY LEVEL. CHECK ACCURACY AND VERIFY BEFORE RUNNING. ---"
echo ""
echo "--- IGNORE THE TEST RESULTS, TO RUN TEST RUN 'cargo test' ---"
echo ""
echo "--- Running benchmark with ${NUM_CORES} threads ---"
echo ""

RAYON_NUM_THREADS="${NUM_CORES}" cargo bench
