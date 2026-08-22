#!/usr/bin/env bash
# 训练完成后自动生成演示视频（mp4 + gif）。
# 用法：
#   bash scripts/make_demo.sh balance        # 手动生成指定任务视频
#   bash scripts/make_demo.sh --auto         # 扫描 5 个任务，为“已完成且无视频”的生成
#   bash scripts/make_demo.sh --auto --force # 忽略已有视频，重新生成
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
# cron 环境无显示会话：用 EGL 无头渲染，保证演示视频可生成
export MUJOCO_GL=egl
TASKS="balance traverse_slope traverse_curve traverse_flat_slope full_chain"
MEDIA="$REPO/media"
mkdir -p "$MEDIA"

make_one() {
  local task="$1" force="${2:-0}"
  local seed="seed00"
  local model="$REPO/rl/runs/$task/$seed/best_model.zip"
  local norm="$REPO/rl/runs/$task/$seed/best_vec_normalize.pkl"
  local final="$REPO/rl/runs/$task/$seed/final_model.zip"
  local comp="$REPO/rl/runs/$task/$seed/.completed"
  local marker="$MEDIA/.demo_${task}_${seed}.done"

  if [ ! -f "$final" ] || [ ! -f "$comp" ]; then
    echo "[$(date '+%H:%M:%S')] 跳过 $task：训练未完成"
    return
  fi
  if [ -f "$marker" ] && [ "$force" = "0" ]; then
    echo "[$(date '+%H:%M:%S')] 跳过 $task：已有演示视频"
    return
  fi
  if [ ! -f "$model" ] || [ ! -f "$norm" ]; then
    echo "[$(date '+%H:%M:%S')] 跳过 $task：缺少 best_model/best_vec_normalize"
    return
  fi

  local extra=""
  [ "$task" = "balance" ] && extra="--disturbance"
  echo "[$(date '+%H:%M:%S')] 生成 $task 演示视频..."
  if ( cd "$REPO" && "$PY" rl/eval.py --task "$task" --model "$model" \
      --vec-norm "$norm" $extra --camera overview --record "$MEDIA/rl_${task}_${seed}" \
      > "$MEDIA/.demo_${task}_${seed}.log" 2>&1 ); then
    # traverse 会按场景输出 rl_<task>_<seed>_<scenario>.*，统一为面板可识别的规范名
    for ext in mp4 gif; do
      for src in "$MEDIA"/rl_"${task}"_"${seed}"_*."${ext}"; do
        [ -f "$src" ] && mv -f "$src" "$MEDIA/rl_${task}_${seed}.${ext}"
      done
    done
    touch "$marker"
    echo "[$(date '+%H:%M:%S')] $task 演示完成"
  else
    echo "[$(date '+%H:%M:%S')] $task 演示失败，见 $MEDIA/.demo_${task}_${seed}.log"
  fi
}

if [ "${1:-}" = "--auto" ]; then
  force=0
  [ "${2:-}" = "--force" ] && force=1
  for task in $TASKS; do
    make_one "$task" "$force"
  done
elif [ $# -eq 1 ]; then
  make_one "$1" 0
else
  echo "用法: $0 <task> | $0 --auto [--force]"
  exit 1
fi
