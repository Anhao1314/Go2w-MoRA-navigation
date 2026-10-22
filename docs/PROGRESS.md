# 进度看板

> **最后更新**：2026-10-23
> **当前阶段**：W1（基线 + C++ 运动学）
> **单一入口**：所有里程碑状态、当前任务、阻塞项都以此页为准；细节见 [项目计划](PLAN.md) 与 [每日日志](journal/)。

## 里程碑

| 里程碑 | 计划日期 | 状态 | 证据卡（产物 + 验证） |
| --- | --- | --- | --- |
| M0 环境 / 协议 / 证据束就绪 | 10/23 | ✅ 完成 | [locomotion/README](../locomotion/README.md)；`python scripts/eval_protocol.py --verify` 全 PASS；2-iteration 冒烟通过 |
| M1 三种子基线 + C++ 运动学 | 10/29 | ✅ 基线达标（运动学第一版已进 CI） | 三份证据束；评估 `vx 0.0569±0.0011` / `ωz 0.0872±0.0017` / `fall 0`；[报告](../reports/locomotion/m1_evaluation.md) |
| M2 C++ 完整仲裁核心 + 集成 | 11/05 | ⏳ 未开始 | — |
| M3 A/B/C 实验 + 归因 | 11/12 | ⏳ 未开始 | — |
| M4 报告 + 演示 | 11/19 | ⏳ 未开始 | — |

## 本周任务（W1）

| ID | 任务 | 状态 | 备注 |
| --- | --- | --- | --- |
| W1.0 | 计划、记录规范、ADR 骨架 | ✅ | [PLAN.md](PLAN.md)、[decisions/](decisions/) |
| W1.1 | S1 三种子训练（seed 0/1/2）+ 评估 | ✅ 完成（1205 / 1232 / 1194 s，全部 rc=0） | 三份证据束（各 31 checkpoints / 37 文件哈希）；三种子评估全部达标，报告见 `reports/locomotion/m1_evaluation.md` |
| W1.1b | 训练记录视频（指标动画） | ✅ | `media/training/m1_training_progress.mp4`（4500 条记录 / 45 秒） |
| W1.1c | 策略行为录像 | ⚠️ 失败，待修 | `render_policy_video.py` 触发 Kit access violation，见 journal |
| W1.2 | Go2 模型参数导出 | ✅ | `locomotion/guardrail/go2_model.json`（19 bodies / 12 joints / 14.971 kg / SHA-256 `12b78af4`） |
| W1.3 | C++ 浮基 FK + CoM + 支撑多边形 | 🟡 第一版完成 | `cpp/guardrail/`（model/kinematics/tests）；CI cpp job 编译与测试通过 |
| W1.4 | 数值对比报告 | ⏳ | 与 Isaac 的 FK/CoM 对比 |
| W1.5 | 命令序列任务环境（P1） | 🟡 第一版完成 | `locomotion/sequence/`，53 维观测，4 阶段状态机，冒烟通过 |

## 阻塞与风险

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 无 | — | 当前无阻塞 |

## 最近产物

| 日期 | 产物 | 路径 |
| --- | --- | --- |
| 10/23 | Go2 刚体模型（C++ 用） | `locomotion/guardrail/go2_model.json` + `.sha256` |
| 10/23 | 实时训练面板（快照 + 视频入口） | `scripts/monitor_training.py` → `http://127.0.0.1:8777` |
| 10/23 | 训练过程 MP4 生成器 | `scripts/render_training_video.py`（已验证可生成） |
| 10/23 | 策略行为 MP4 录制器 | `scripts/render_policy_video.py`（待训练结束后验证） |
| 10/23 | 架构 v0.2 提案（C++ 仲裁层） | `docs/ARCHITECTURE_V2.md` |
| 10/23 | 执行计划 + 记录规范 | `docs/PLAN.md` |
| 10/23 | 每 run 证据束（自动） | `logs/rsl_rl/go2_loco_s1_flat/<stamp>/run_config.json` |
| 10/23 | 2048 envs 吞吐基准 | `docs/ROADMAP.md` §3（34.36 s / 20 iters，峰值 4208 MiB） |

## 使用约定

1. 每完成一项任务：更新本页对应行 + 写当日 journal + 提交推送。
2. 每完成一个里程碑：在"证据卡"列补一条可复现命令与产物路径。
3. 有阻塞：立即写入"阻塞与风险"，并记录在当天 journal。
