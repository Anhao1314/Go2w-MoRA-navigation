# 实验数据记录与论文证据协议

本文定义一次 Go2W/MoRA-Train 训练 run 需要保存什么、每种数据能支持什么结论，以及如何在不改写历史的前提下生成证据清单。

## 1. 一次 run 的目录

```text
rl/runs/<task>/<job-id>/
  run_config.json              # 配置、seed、依赖版本、输入文件哈希
  events.jsonl                 # v0.2 训练事件流
  stage1/
    eval_log.csv               # 每 50k 步的聚合评估
    best_model.zip
    best_vec_normalize.pkl
    final_model.zip
    final_vec_normalize.pkl
    tensorboard/               # SB3 训练曲线
    .completed
  stage2/ ...
  stage3/ ...
  evaluation/                  # 最终 20-episode 验收
    summary.json
    episodes.json
    metrics.csv
    report.md
  _job/                        # fetch 时复制
    spec.json                  # 命令、env、Mac 源提交
    meta.json                  # 启动/结束时间、PID、exit code
    train.log
  evidence_manifest.json       # collector 生成，SHA-256 清单
```

## 2. 每类数据的用途

| 数据 | 用途 | 证据口径 |
| --- | --- | --- |
| `run_config.json` | 复现配置、seed、依赖版本、BC 输入哈希 | 运行开始时创建，不覆盖 |
| `events.jsonl` | as-of 训练分析、早停规则、跨 run 因子 | `event_time`/`available_at` 时间合同 |
| `stage*/eval_log.csv` | 学习曲线、阶段成功率、论文折线图 | 离线聚合评估，不含每行可用时间 |
| `stage*/tensorboard/` | PPO loss/KL/entropy 等训练曲线 | 训练过程日志，不是评估标签 |
| `stage*/best_model.zip` | 可回放的策略 checkpoint | 与 `best_vec_normalize.pkl` 成对使用 |
| `evaluation/summary.json` | 最终 verdict、success_rate、checkpoint 哈希 | 最终验收，不是训练中指标 |
| `evaluation/episodes.json` | 每个验收 episode 的 seed、成功、步数、时间 | 最终 20 episodes |
| `_job/spec.json` | 训练命令、环境变量、Mac 源提交 | 解决 Windows 工作区无 `.git` 的 provenance 问题 |
| `_job/meta.json` | 启动/结束时间、PID、exit code | 墙钟时间，不是 GPU 时间 |
| `evidence_manifest.json` | 文件哈希、事件校验、阶段完成度 | 只哈希已有文件，不改写 |

## 3. 论文/报告建议使用方式

- **训练曲线**：使用 `stage*/eval_log.csv` 或 TensorBoard；图中注明固定场景、无域随机化。
- **最终结果表**：使用 `evaluation/metrics.csv` 和 `evaluation/episodes.json`，同时报告 20 episodes 的分母。
- **训练可靠性**：把 `events.jsonl` 导入 RL-Sentinel，做 as-of replay；不要把 `eval_log.csv` 伪装成带有可用时间的事件。
- **可复现性**：报告 `run_config.json` 中的 seed、输入哈希、依赖版本，以及 `_job/spec.json` 的源提交和 dirty 状态。
- **策略回放**：checkpoint 与匹配的 `VecNormalize` 必须成对保存；不能拿 BC 权重或旧归一化统计代替。
- **失败 run**：保留失败/中断 run，label 保持 `unknown` 或 `fail`；不要删除或事后补时间。

## 4. 收集流程

训练完成后，在 Mac 上：

```bash
python tools/win_train.py fetch mseg-events-full-s0b
```

```bash
python scripts/collect_run_evidence.py \
  --run-dir rl/runs/_fetched/mseg-events-full-s0b \
  --bundle-dir reports/evidence/mseg-events-full-s0b
```

`fetch` 会拉取完整 run 目录和 `_job/spec.json`、`meta.json`、`train.log`。`collect_run_evidence.py` 会：

1. 校验 `events.jsonl` 的 schema、时间顺序、单 start、finish/label 顺序、sample step 单调性；
2. 汇总每个 stage 的 `eval_log.csv` 行数、最后一行、checkpoint 和 TensorBoard 哈希；
3. 写入 `evidence_manifest.json`（不覆盖已有文件）；
4. 可选生成紧凑 bundle：config、events、eval CSV、最终验收、job provenance、manifest；checkpoint 留在完整 run 目录并以哈希登记。

`--allow-incomplete` 只用于检查训练中的 run，不用于论文最终证据。

## 5. 当前 run 的记录状态

`mseg-events-full-s0b`（2026-10-19，已完成）：

- 任务：multi-segment，seed 0，CPU，计划 2M steps；3 个 curriculum stage 全部完成并提前停止，事件步 900k。
- 已归档：`reports/evidence/mseg-events-full-s0b/`（120 KB），含 `run_config.json`、`events.jsonl`、3 个 stage 的 `eval_log.csv`、`evaluation/summary.json`、`episodes.json`、`metrics.csv`、`_job/spec.json`、`train.log` 和 `bundle_manifest.json`。
- 最终验收：20 episodes，success_rate 1.0，mean_x 10.6635，mean_ep_len 1428，label `pass`。
- 源提交：`_job/spec.json` 记录 `c599d7f`，dirty 1（`terrain.STL` 大小写幻影，不进入提交）。
- 事件流：每 5000 步一个 sample；最终验收后写 `finish` 和 `pass/fail/unknown` label。
- 目前 `run_config.runtime.source_commit` 仍为 null（该 run 启动于 provenance env 注入之前）；以 `_job/spec.json` 为准。后续 run 会直接写入 `run_config.runtime`。

## 6. 尚未记录、后续补充

| 缺口 | 影响 | 计划 |
| --- | --- | --- |
| 奖励分量 | 无法做 reward shaping 归因 | 在 `info` 中增加分量并在 observer 中记录 |
| 每步完整轨迹 | 只能看聚合曲线和最终 episodes | 对关键 episode 保存 `(x,y,yaw,step,action)` NPZ/CSV |
| stage eval 的每 episode 明细 | 只有聚合成功率 | 评估 callback 保存 per-episode JSON |
| 默认视频 | 论文演示需要单独录制 | 训练后用 best checkpoint 跑独立 recorder |
| 评估指标的严格可用时间 | `eval_log.csv` 是离线表 | 在评估 callback 中写带 `available_at` 的 eval 事件 |

这些缺口不会被“补造时间”或“用最终结果反推训练过程”掩盖；需要新 run 采集就重新跑，不修改旧证据。

## 7. 论文证据清单

- [ ] `run_config.json` + `_job/spec.json`：配置、seed、源提交、依赖版本
- [ ] `events.jsonl`：start/sample/finish/label，step 单调，时间合法
- [ ] `stage*/eval_log.csv` 或 TensorBoard：训练曲线
- [ ] `evaluation/summary.json` + `episodes.json`：最终结果与分母
- [ ] checkpoint + `VecNormalize`：可回放
- [ ] `evidence_manifest.json`（完整 run）或 `bundle_manifest.json`（紧凑归档）：SHA-256 清单
- [ ] 明确写出 fixed-scene / no-domain-randomization / simulation-only 边界
- [ ] 失败 run 与 abstain 不删除、不重写
