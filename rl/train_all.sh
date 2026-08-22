#!/usr/bin/env bash
# 完整训练预算：Phase1（balance）与 Phase2（full_chain）各 3 个随机种子。
# 可用环境变量 STEPS 覆盖步数，默认 8000000。
set -euo pipefail

STEPS="${STEPS:-8000000}"

for seed in 0 1 2; do
  python rl/train.py --task balance --total-steps "$STEPS" --seed "$seed"
done

for seed in 0 1 2; do
  python rl/train.py --task full_chain --total-steps "$STEPS" --seed "$seed" \
    --init-from "rl/runs/balance/seed$(printf '%02d' "$seed")/best_model.zip"
done
