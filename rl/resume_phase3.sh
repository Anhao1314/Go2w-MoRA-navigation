#!/usr/bin/env bash
# 从断点继续 Phase 3：traverse seed0 续训，seed1/2 新训。
set -euo pipefail

STEPS="${STEPS:-8000000}"

if systemctl --user is-active --quiet go2w-train-guard; then
  echo "错误：go2w-train-guard 正在运行，请先停止守护再手动执行本脚本" >&2
  exit 1
fi

if [ -f rl/runs/traverse/seed00/best_model.zip ]; then
  python rl/train.py --task traverse --total-steps "$STEPS" --seed 0 \
    --resume-from rl/runs/traverse/seed00/best_model.zip \
    --resume-norm rl/runs/traverse/seed00/best_vec_normalize.pkl
else
  python rl/train.py --task traverse --total-steps "$STEPS" --seed 0
fi

for seed in 1 2; do
  python rl/train.py --task traverse --total-steps "$STEPS" --seed "$seed"
done
