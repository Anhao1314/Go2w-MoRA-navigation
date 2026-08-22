#!/usr/bin/env bash
# 从断点继续 Phase 1/2：balance seed0 续训，seed1/2 新训，然后 full_chain 3 seeds。
set -euo pipefail

STEPS="${STEPS:-8000000}"

if systemctl --user is-active --quiet go2w-train-guard; then
  echo "错误：go2w-train-guard 正在运行，请先停止守护再手动执行本脚本" >&2
  exit 1
fi

if [ -f rl/runs/balance/seed00/best_model.zip ]; then
  python rl/train.py --task balance --total-steps "$STEPS" --seed 0 \
    --resume-from rl/runs/balance/seed00/best_model.zip \
    --resume-norm rl/runs/balance/seed00/best_vec_normalize.pkl
else
  python rl/train.py --task balance --total-steps "$STEPS" --seed 0
fi

for seed in 1 2; do
  python rl/train.py --task balance --total-steps "$STEPS" --seed "$seed"
done

for seed in 0 1 2; do
  python rl/train.py --task full_chain --total-steps "$STEPS" --seed "$seed" \
    --init-from "rl/runs/balance/seed$(printf '%02d' "$seed")/best_model.zip"
done
