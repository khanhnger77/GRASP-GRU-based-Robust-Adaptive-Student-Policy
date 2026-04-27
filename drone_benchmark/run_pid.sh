#!/usr/bin/env bash
# PID full benchmark: baseline tests + payload tests.
# Payload drop is the final scene.
#
# Baseline SITL (gz_x500, flat world):
#   cd embedded_platforms/px4/px4_autopilot
#   make px4_sitl gz_x500 \
#       EXTERNAL_MODULES_LOCATION="$(cd .. && cd external_modules && pwd)" \
#       RL_TOOLS_ROOT="$(cd ../../../ && pwd)" -j8
#
# Payload SITL (gz_x500_payload_drop_lawn):
#   make px4_sitl gz_x500_payload_drop_lawn \
#       EXTERNAL_MODULES_LOCATION="$(cd .. && cd external_modules && pwd)" \
#       RL_TOOLS_ROOT="$(cd ../../../ && pwd)" -j8

set -e
cd "$(dirname "$0")"

CTRL="pid"
WAIT=8

echo "=========================================================="
echo " PID Full Benchmark"
echo "=========================================================="

# ==========================================================================
# PART 1 — Baseline tests (gz_x500, flat world, no payload)
# ==========================================================================
echo ""
echo "────────────────────────────────────────────────────────"
echo " PART 1: Baseline tests  (model: gz_x500)"
echo "────────────────────────────────────────────────────────"
read -r -p " SITL running with gz_x500? Press Enter to start..."
echo ""

for TEST in step figure8 hover disturbance; do
    echo ""
    echo ">>> TEST: $TEST  (controller=$CTRL)"
    python main.py --test "$TEST" --controller "$CTRL"
    echo "<<< Done: $TEST"
    echo "    Waiting ${WAIT}s..."
    sleep "$WAIT"
done

echo ""
echo " Part 1 complete. Logs → logs/   Plots → plots/"

# ==========================================================================
# PART 2 — Payload hover  (gz_x500_payload_drop_lawn)
# ==========================================================================
echo ""
echo "────────────────────────────────────────────────────────"
echo " PART 2: Payload hover  (model: gz_x500_payload_drop_lawn)"
echo "────────────────────────────────────────────────────────"
echo ""
echo " Restart SITL with the payload model."
echo ""
read -r -p " SITL running with gz_x500_payload_drop_lawn? Press Enter to start payload_hover..."
echo ""
python main.py --test payload_hover --controller "$CTRL"
echo "<<< Done: payload_hover"
echo "    Waiting ${WAIT}s..."
sleep "$WAIT"

# ==========================================================================
# PART 3 — Payload drop (FINAL SCENE)  (gz_x500_payload_drop_lawn)
# ==========================================================================
echo ""
echo "────────────────────────────────────────────────────────"
echo " PART 3: Payload drop — FINAL SCENE"
echo "         (model: gz_x500_payload_drop_lawn)"
echo "────────────────────────────────────────────────────────"
echo ""
echo " Restart SITL (payload must be attached at sim start)."
echo ""
read -r -p " SITL running with payload attached? Press Enter to start payload_drop..."
echo ""
python main.py --test payload_drop --controller "$CTRL"
echo "<<< Done: payload_drop"

echo ""
echo "=========================================================="
echo " PID Full Benchmark complete."
echo " Logs   → logs/"
echo " Plots  → plots/<test_name>/"
echo ""
echo " Generate videos:"
echo "   python make_video.py logs/ --out videos/"
echo "=========================================================="
