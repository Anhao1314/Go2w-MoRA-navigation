# 架构 v0.2：可验证仲裁层（Verifiable Arbitration Layer, VAL）

[返回首页](../README.md) · [文档索引](README.md) · [MoRA 系统层](MORA_TRAIN.md) · [实验协议](EXPERIMENT_DESIGN.md)

> **状态**：架构提案 + 研究计划（2026-10-23）。取代此前以纯运动控制性能为导向的 [MVP.md](MVP.md)。

## 1. 为什么走架构研究

纯 RL 训练 Go2 走路在同质化竞争中很难形成辨识度；本仓库真正的独特资产是**分层架构 + 可验证控制**：

- `mora/` 已有 S2 规划 → S1 策略 → memory/progress 的运行时闭环
- `cpp/` 有 MPC/FK/IK 的数值验证栈
- Phase-1 已交付事后停机路径（M2.1 physical halt：速度从 0.451 降到 0.056 m/s）

但 v0.1 的层间只有**数据契约**（`Goal` / `Action`），没有**约束契约**：S1 输出的任何动作都直接进入 S0，安全性依赖训练分布和事后停机。这正是可以研究并改进的架构缺口。

**v0.2 的主张**：分层具身系统的可靠性不应只来自"训练得更稳"，也应来自**层间的可验证约束**——让失败可拦截、干预可量化、行为可归因。

## 2. 架构

```text
S2 Planner（规则 / 未来可换 VLM）
   │  结构化 Goal
   ▼
S1 Policy（PPO / BC / 未来 VLA）
   │  ActionProposal（原始动作）
   ▼
┌─────────────────────────────────────┐
│  Arbiter（可验证仲裁层，新增）        │
│  输入：observation + proposal         │
│  规则：姿态包络 / 关节速度 / 扭矩饱和 │
│        / 基座高度 / 进度停滞          │
│  输出：ALLOW / CLAMP / FALLBACK / HALT│
└─────────────────────────────────────┘
   │  safe action + 触发记录
   ▼
S0 Controller / Actuator（PD / MPC / 学习型策略）
   │
   ▼
Environment ──────────────► observation
```

**与 v0.1 的差异**：多了一个"约束契约"层。它不学习、不训练，只做可验证的规则检查——因此可以被单元测试覆盖，也可以在报告里逐条说明触发条件。

**对现有代码的侵入**：`AgentRuntime.step()` 在 `policy.act()` 与 `task.step()` 之间插入 `arbiter.review()`。默认仲裁器为 `AllowAllArbiter`，行为与 v0.1 完全一致，向后兼容。

## 3. 层间契约

```python
@dataclass(frozen=True)
class ActionProposal:
    action: Action
    source: str          # "policy" | "planner" | "fallback"
    step: int

@dataclass(frozen=True)
class ArbitrationDecision:
    verdict: Literal["ALLOW", "CLAMP", "FALLBACK", "HALT"]
    action: Action            # 实际转发给 S0 的动作
    triggers: tuple[str, ...] # 触发的规则名，按严重度排序
    margins: Mapping[str, float]  # 各规则距阈值的余量（用于分析）

class Arbiter(Protocol):
    def reset(self, goal: Goal) -> None: ...
    def review(self, observation: Observation, proposal: ActionProposal) -> ArbitrationDecision: ...
```

**可验证性要求**（这是"可验证"三个字的实际含义）：

1. 每条规则是纯函数：`(observation, action) -> (triggered, margin)`；
2. 所有输入来自 observation/info，不依赖隐藏状态；
3. 每条规则必须有边界单元测试（恰好超限、恰好未超限）；
4. `CLAMP` 后的动作必须仍在安全包络内（可断言）。

## 4. 规则集（初版）

| 规则 | 观测量 | 触发 | 裁决 | 可验证点 |
| --- | --- | --- | --- | --- |
| `attitude_envelope` | roll / pitch | 超出包络 | CLAMP | 裁剪后角度 ≤ 限值 |
| `joint_velocity` | 12 关节速度 | 超限 | CLAMP | 裁剪后速度 ≤ 限值 |
| `torque_saturation` | 连续饱和步数 | ≥ N 步 | FALLBACK | 计数状态可复现 |
| `base_height` | 基座高度 | 低于阈值 | FALLBACK → HALT | 高度来自仿真状态 |
| `progress_stall` | progress 停滞步数 | ≥ M 步 | HALT | 与现有 ProgressEstimator 复用 |

阈值来源：**从基线策略的统计分布取分位数**（例如 roll 的 99.5 分位），而不是凭感觉定。这本身是可讲的工程细节。

## 5. 研究问题与假设

**RQ1（有效性）**：仲裁层能否在不显著降低任务成功率的前提下，降低危险状态发生率？

> H1：相对无仲裁基线，危险状态率下降 ≥ 50%，成功率下降 ≤ 5%。若成功率损失过大，则报告"可验证约束的代价"曲线——负结果同样可用。

**RQ2（权衡）**：干预率与性能损失的 Pareto 前沿是什么？固定阈值与自适应阈值（按速度指令/地形缩放）哪个更优？

> H2：固定阈值在平地更优；自适应阈值在扰动/轻地形更优。

**RQ3（可归因性）**：仲裁触发分布能否解释策略的失败模式？

> H3：多数危险动作在触发前已有可观测征兆（姿态角速度上升），即"可预判"。

## 6. 实验设计

| 配置 | 说明 |
| --- | --- |
| **A（基线）** | 无仲裁（v0.1 行为） |
| **B（固定阈值）** | 规则阈值取基线分布的 99.5 分位 |
| **C（自适应阈值）** | 阈值随命令速度/地形难度缩放 |

| 场景 | 环境 | 目的 |
| --- | --- | --- |
| S1 平地 | `Go2Loco-S1-Flat-Blind-v0` | 基本有效性与性能损失 |
| S2 扰动 | `Go2Loco-S2-Disturbance-v0` | 干预是否集中于扰动时刻 |
| S2b 轻地形 | `Go2Loco-S2b-LightTerrain-v0` | 泛化到地形变化 |

- 评估种子：冻结集 `E={0..19}`，每配置 20 episodes；
- 策略：S1 三种子（P0 产出），A/B/C 共用同一策略权重，**只改仲裁层**；
- 指标：危险状态率（摔倒 + 姿态超限）、成功率、干预率、平均奖励、触发原因分布、恢复时间。

## 7. 交付物

1. **架构文档**（本文）+ 层间契约与状态机图
2. **实现**：`mora/arbitration.py`（契约 + 规则 + AllowAllArbiter）、`locomotion/guardrail.py`（Go2 规则绑定）
3. **单元测试**：每条规则的边界用例 + CLAMP 后安全性断言
4. **实验**：A/B/C × 3 场景 × 20 episodes，含触发归因
5. **报告**：`reports/guardrail/`，含证据束（run_config + SHA-256）
6. **演示**：同一 episode 在有/无仲裁下的对比视频（含触发时刻标注）
7. **README 首屏**：架构图 + 一句话结论

## 8. 时间盒（3 周 + 1 周包装）

| 周 | 内容 | 里程碑 |
| --- | --- | --- |
| W1 | P0 三种子训练（2.2 h）+ 仲裁契约/规则实现（纯 Python）+ 单元测试 | 三种子证据束；规则边界测试通过 |
| W2 | Go2 集成、阈值标定（基线分位数）、A 与 B 跑通 | A/B 在 S1 上各 20 episodes |
| W3 | C 配置 + 三场景完整实验 + 触发归因 | A/B/C 结果表 + 主因分布 |
| W4 | 报告、README、视频、演练 | 可复现 + 可讲 |

## 9. 成功判据

**必须：**
- [ ] 契约实现且向后兼容（默认 AllowAllArbiter 不改变 v0.1 行为）
- [ ] 每条规则有边界单元测试，CLAMP 后安全性可断言
- [ ] A/B/C × 3 场景 × 20 episodes 完整
- [ ] 报告明确回答 RQ1–RQ3（成立或不成立并给出解释）
- [ ] 阈值标定方法可复现（分位数来源）
- [ ] 证据束 + 一键命令 + CI 绿

**加分：** 触发记录能定位到具体失败模式，并给出"哪类状态应加入训练分布"的建议（闭环回训练）。

## 10. 边界与风险

**不做**：新 RL 算法、真机、VLA、S3 楼梯、多机。

| 风险 | 预案 |
| --- | --- |
| 仲裁频繁触发导致成功率大跌 | 报告 Pareto 曲线；阈值上调至 99.9 分位；把"性能损失"作为结果而非失败 |
| 阈值取自分位数导致对扰动场景过严 | C 配置按场景缩放；报告中给出两套阈值 |
| 仲裁掩盖策略缺陷 | 报告区分"被拦截的危险动作"与"策略本就不会做"；对比 A 的失败样本 |
| 规则实现与仿真状态不同步 | 规则只读 observation/info，不读仿真内部对象 |

## 11. 与 Phase-1 的关系

Phase-1 的 M2.1 证明了"事后物理停机"可行（失败后 StopSkill 降速）。v0.2 把安全机制前移为**事前仲裁**，并将它从"某个 adapter 的配置项"提升为"架构层契约"。两者构成同一条演进线：

```text
事后停机（M2.1） → 事前仲裁（v0.2） → 形式化安全过滤器（未来，MPC barrier）
```
