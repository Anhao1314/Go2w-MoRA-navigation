#!/usr/bin/env bash
# Phase 3：场景穿越（traverse）完整训练预算，3 个随机种子。
# 可用环境变量 STEPS 覆盖步数，默认 8000000。
set -euo pipefail

STEPS="${STEPS:-8000000}"

for seed in 0 1 2; do
  python rl/train.py --task traverse --total-steps "$STEPS" --seed "$seed"
done
