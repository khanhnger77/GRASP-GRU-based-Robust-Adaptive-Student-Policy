#!/usr/bin/env bash
# Run from the repo root:
#   bash src/payload_policy/post_training/build_actor.sh
g++ -std=c++17 src/payload_policy/post_training/load_actor.cpp -I include && ./a.out
