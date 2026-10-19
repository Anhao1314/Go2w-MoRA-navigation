# MoRA-Train 系统层（v0.1 骨架）

本文记录 `mora/` 包的当前能力、接口和边界。它是 MoRA-Train 的系统层，不是 Go2W 环境本身的说明；Go2W 只是第一个 task adapter。

## 1. 当前状态

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| Task Adapter API | ✅ 已实现 | `reset / step / goal / progress / success / metadata / replan` |
| Agent Runtime | ✅ 已实现 | S2 planner → S1 policy → task step → memory/progress 闭环 |
| S2 Planner | ✅ 规则版 | `RulePlanner` 使用 adapter 的当前目标；停滞时调用 `replan()` |
| S1 Policy | ✅ 接口 + 轻量实现 | `CallablePolicy`、`ZeroPolicy`、`SB3Policy` |
| 短期记忆 | ✅ 已实现 | 固定容量 recent transition buffer |
| 情节记忆 | ✅ 已实现 | 已完成 episode 的 summary，有容量上限 |
| 任务记忆 | ✅ 已实现 | 跨 episode 持久 key/value；不随 episode reset 清空 |
| 神经 Memory Token | ⏳ 未实现 | 当前是显式、可检查的记忆快照，不是官方 MoRA 的 token 实现 |
| 进度估计 | ✅ 启发式 | EMA delta + stall counter + replan signal |
| 学习型 progress head | ⏳ 未实现 | 接口已留出，后续用辅助任务训练 |
| 事件流 | ✅ 已实现 | v0.2 风格 `start / sample / finish / label`，强制 `event_time <= available_at` |
| SB3 observer | ✅ 已实现 | `Sb3EventCallback` 周期性写 sample，训练结束写 finish |
| Go2W adapter | ✅ 已实现 | multi-segment / junction / high-level-curve |
| 训练脚本接入 | ⏳ 下一步 | 目前通过 adapter/runtime 可运行，但尚未替换四个训练入口 |
| RL-Sentinel ingester | ⏳ 下一步 | Mac 侧尚未消费 Go2W 的事件扩展字段 |

测试：`249 tests OK (1 skipped)`（2026-10-19，macOS），`mora` 通过 Pyright。

## 2. 运行时循环

```text
TaskAdapter.reset()
  → Planner.reset()
  → loop:
      goal = Planner.decide(task, memory, progress)
      action = Policy.act(observation, goal, memory.snapshot())
      observation, reward, done, info = TaskAdapter.step(action)
      progress = ProgressEstimator.update(info["progress"])
      memory.observe(Transition(...))
      if progress.should_replan: Planner 记录 replan 请求
      if done: memory.record_episode(EpisodeSummary(...))
```

关键设计：
- **环境不拥有记忆**：记忆在 runtime 内，环境只提供观测、进度和终止。
- **S2 不直接输出动作**：S2 输出结构化 `Goal`，S1 负责执行。
- **进度闭环可替换**：`ProgressEstimator` 是 Protocol，后续可换成学习型 progress head。
- **事件与决策分离**：`EventWriter` 只记录，不影响训练/评估。

## 3. Task Adapter API

```python
class TaskAdapter(Protocol):
    observation_space: Any
    action_space: Any

    def reset(seed=None, options=None) -> (observation, info): ...
    def step(action) -> (observation, reward, terminated, truncated, info): ...
    def goal() -> Goal: ...
    def progress() -> float: ...
    def success() -> bool: ...
    def metadata() -> dict: ...
    def replan() -> Goal: ...
    def close() -> None: ...
```

Go2W 的实现见 [`mora/adapters/go2w.py`](../mora/adapters/go2w.py)：

```python
from mora.adapters.go2w import Go2wTaskAdapter

adapter = Go2wTaskAdapter("multi-segment", scope=2, s0="scripted")
observation, info = adapter.reset(seed=0)
goal = adapter.goal()          # Goal(kind="subgoal", label="segment-...")
```

## 4. 最小运行示例

```python
from mora.adapters.go2w import Go2wTaskAdapter
from mora.policy import ZeroPolicy
from mora.runtime import AgentRuntime

task = Go2wTaskAdapter("multi-segment", scope=2)
runtime = AgentRuntime(task, ZeroPolicy(action_dim=2), max_steps=50)
result = runtime.run_episode(seed=0, max_steps=50)
print(result)
runtime.close()
```

`ZeroPolicy` 只用于接口验证；真实训练时用 `SB3Policy(model)` 或自定义策略。

## 5. 事件合同

`EventWriter` 写入 JSONL，首行是：

```json
{"schema_version":2,"origin":"observed"}
```

之后每行一个事件：

```json
{"schema_version":2,"event_id":"...","run_id":"...","kind":"sample",
 "event_time":..., "available_at":..., "payload":{"step":..., "reward":...}}
```

硬约束：
- `event_time <= available_at`
- `start` 只能一次；`sample` 必须在 active run 中；`finish` 只能一次；`label` 必须在 `finish` 之后
- NaN/Infinity 拒绝；文件独占创建，避免覆盖其他 run 的证据
- `Sb3EventCallback` 只写 `start/sample/finish`；最终 `pass/fail` 由验收评估追加

当前 sample 扩展字段包括 `progress`、`success_rate`、`segment`。RL-Sentinel 的 Go2W ingester 需要显式接受这些字段，而不是把它们当作旧 schema 的未知字段。

## 6. 与 Go2W 训练的关系

- 当前 Go2W 环境仍由 `HighLevelEnvWrapper` 提供观测和任务进度。
- `Go2wTaskAdapter` 复用 `make_stage_env` / `make_high_level_env`，保证 reward、termination、观测语义与 P0 一致。
- C++ S0 目前只接到 multi-segment 的 `s0="cpp"`；junction/high-level-curve 仍用 scripted S0。
- 训练脚本尚未默认使用 `AgentRuntime`；下一步是给训练脚本加 `--event-log` / observer，并在评估后写 `label`。

## 7. 下一步

1. 把 `Sb3EventCallback` 接入 `train_multi_segment_curriculum.py` 等入口，生成真实训练事件。
2. 在 RL-Sentinel 增加 Go2W ingester，接受 `progress/success_rate/segment` 扩展字段。
3. 用 `AgentRuntime` 跑一条完整 multi-segment episode，对比 scripted 路径的 success/progress。
4. 加一个学习型 progress head 和一个记忆消融实验。
5. 再考虑 ROS2/S0 500 Hz 节点。

## 8. 边界

- **MoRA-inspired**，不是官方 MoRA 复现；官方没有公开代码/论文。
- 当前记忆是显式数据结构，不是神经 Memory Token；不声称等价。
- 当前 progress 是启发式信号，不是校准后的失败概率。
- 仿真-only；无实机、无 sim-to-real。
- 249 tests 是全仓库 Python 回归，不等于 MoRA-Train 整体已被验证。
