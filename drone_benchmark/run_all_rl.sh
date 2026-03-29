#!/usr/bin/env bash
# Run all 4 benchmark tests with RL controller sequentially.
# Land + disarm between each test, then wait for drone to fully settle.

set -e
cd "$(dirname "$0")"

CTRL="rl"
WAIT=8   # seconds between tests for drone to land and reset

echo "=================================================="
echo " RL Benchmark — all tests"
echo "=================================================="

for TEST in hover step figure8 disturbance; do
    echo ""
    echo ">>> TEST: $TEST  (controller=$CTRL)"
    python main.py --test "$TEST" --controller "$CTRL"
    echo "<<< Done: $TEST"
    echo "    Waiting ${WAIT}s before next test..."
    sleep "$WAIT"
done

echo ""
echo "=================================================="
echo " All RL tests complete."
echo " Logs  → logs/"
echo " Plots → plots/"
echo "=================================================="
