# 架构 v0.2：可验证仲裁层（Verifiable Arbitration Layer, VAL）

[返回首页](../README.md) · [文档索引](README.md) · [MoRA 系统层](MORA_TRAIN.md) · [实验协议](EXPERIMENT_DESIGN.md)

> **状态**：架构提案 + 研究计划（2026-10-23）。取代以纯运动控制性能为导向的 [MVP.md](MVP.md)。

## 1. 为什么走架构研究

纯 RL 训练 Go2 走路在同质化竞争中很难形成辨识度；本仓库真正的独特资产是**分层架构 + 可验证控制**：

- `mora/` 已有 S2 规划 → S1 策略 → memory/progress 的运行时闭环
- `cpp/` 有浮基 FK/IK、质心动力学、SRBD MPC（OSQP）与安全状态机，均有数值验证
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
┌──────────────────────────────────────────────┐
│  Arbiter（可验证仲裁层，C++ 核心）             │
│  运动学：浮基 FK、足端位置、支撑多边形          │
│  动力学：CoM/ZMP、摩擦锥、RNID 力矩可行性       │
│  约束：关节位置/速度/力矩限位                  │
│  输出：ALLOW / CLAMP / FALLBACK / HALT        │
└──────────────────────────────────────────────┘
   │  safe action + 触发记录（规则名 + 余量）
   ▼
S0 Controller / Actuator（PD / MPC / 学习型策略）
   │
   ▼
Environment ──────────────► observation
```

**与 v0.1 的差异**：多了一个"约束契约"层。它不学习、不训练，只做可验证的物理检查——因此可以被单元测试覆盖，也可以逐条说明触发条件。

**对现有代码的侵入**：`AgentRuntime.step()` 在 `policy.act()` 与 `task.step()` 之间插入 `arbiter.review()`。默认仲裁器为 `AllowAllArbiter`，行为与 v0.1 完全一致，向后兼容。

## 3. 技术栈（明确的分工）

| 层 | 语言 / 工具 | 组件 | 职责 |
| --- | --- | --- | --- |
| 编排层 | Python 3.11 | `mora/arbitration.py` | 契约、状态机、策略集成、触发记录 |
| **仲裁核心** | **C++17** | `cpp/guardrail/` | 运动学/动力学检查、批量裁决 |
| 绑定 | pybind11 | `cpp/bindings/guardrail_py.cpp` | numpy ↔ Eigen 批量接口 |
| 验证 | Catch2 + Python | `cpp/tests/test_guardrail_*.cpp` | 数值一致性、边界用例 |
| 仿真/训练 | Python | Isaac Lab 2.7.0 | 策略训练与评估 |
| 评估 | Python | `scripts/eval_guardrail.py` | A/B/C 对比、归因统计 |

### 3.1 运动物理学（仲裁核心的数学基础）

仲裁不是"拍脑袋的阈值"，而是基于刚体动力学与接触力学：

| 检查 | 物理内容 | 触发含义 |
| --- | --- | --- |
| **支撑多边形 / CoM** | 浮基 FK → 足端接触点 → 凸包；质心投影是否落在支撑多边形内（准静态稳定性判据） | 即将失稳（倾倒趋势） |
| **摩擦锥** | 由期望加速度估算地面反力，检查 `μ·Fz ≥ √(Fx²+Fy²)` | 打滑风险 |
| **RNID 力矩可行性** | 由 `q, q̇, q̈` 逆动力学计算所需关节力矩，与执行器限幅比较 | 执行器能力不足（跟踪不上/饱和） |
| **关节约束** | 位置、速度、力矩限位（来自 URDF/仿真参数） | 越限 |
| **进度停滞** | 复用 `ProgressEstimator` | 任务层失败 |

其中 CoM/支撑多边形与摩擦锥是足式机器人运动控制的经典判据；RNID 提供动态可行性。三者组合能覆盖"静不稳、打滑、执行器饱和"三类主要失败模式。

### 3.2 为什么必须用 C++

1. **实时性**：控制频率 50 Hz、并行 2048 环境。Python 逐环境循环的调用开销不可接受；C++ 批量接口（numpy 进、numpy 出）目标 **< 1 ms / step（2048 envs）**，并有基准测试。
2. **可移植**：未来上真机时，C++ 核心可直接编译为 ROS2 节点，不需要重写物理检查。
3. **可验证**：Catch2 单元测试覆盖边界条件；数值结果与 Isaac Lab 的 FK/CoM 逐项对比（误差目标：位置 < 1e-4 m，力矩 < 1e-2 N·m）。
4. **复用**：`cpp/` 已有浮基 FK、质心动力学与 Eigen 依赖；新模块独立编译，**不修改 Phase-1 冻结的 Go2W 栈**。

### 3.3 Go2 模型参数来源

Go2 的连杆质量/惯量/关节限位来自 Isaac Lab 的官方资产（`Nucleus/Robots/Unitree/Go2/go2.usd`）。流程：

1. 在 Kit 中加载 Go2，导出每个连杆的 mass / inertial / joint axis / limits 为 JSON（`scripts/export_go2_model.py`）；
2. C++ 侧启动时加载该 JSON（不硬编码）；
3. 导出的参数与 Isaac 运行时数据做一致性测试，防止模型漂移。

这样 C++ 侧的物理检查与仿真使用同一套参数，结论可追溯。

## 4. 层间契约

```python
@dataclass(frozen=True)
class ActionProposal:
    action: Action
    source: str          # "policy" | "planner" | "fallback"
    step: int

@dataclass(frozen=True)
class ArbitrationDecision:
    verdict: Literal["ALLOW", "CLAMP", "FALLBACK", "HALT"]
    action: Action                 # 实际转发给 S0 的动作
    triggers: tuple[str, ...]      # 触发的检查名，按严重度排序
    margins: Mapping[str, float]   # 各检查距阈值的余量（用于分析）

class Arbiter(Protocol):
    def reset(self, goal: Goal) -> None: ...
    def review(self, observation: Observation, proposal: ActionProposal) -> ArbitrationDecision: ...
```

**可验证性要求**（"可验证"的实际含义）：

1. 每个检查是纯函数：`(state, action) -> (triggered, margin)`；
2. 输入只来自 observation/info 与模型参数，不读仿真内部对象；
3. 每个检查有边界单元测试（恰好超限、恰好未超限）；
4. `CLAMP` 后的动作必须满足约束（断言）；
5. C++ 核心与 Python 参考实现的结果一致（同一组随机输入下逐项对比）。

## 5. 研究问题与假设

**RQ1（有效性）**：仲裁层能否在不显著降低任务成功率的前提下，降低危险状态发生率？

> H1：相对无仲裁基线，危险状态率下降 ≥ 50%，成功率下降 ≤ 5%。若成功率损失过大，则报告"可验证约束的代价"曲线——负结果同样可用。

**RQ2（权衡）**：干预率与性能损失的 Pareto 前沿是什么？固定阈值与自适应阈值（按速度指令/地形缩放）哪个更优？

> H2：固定阈值在平地更优；自适应阈值在扰动/轻地形更优。

**RQ3（可归因性）**：仲裁触发分布能否解释策略的失败模式？

> H3：多数危险动作在触发前已有可观测征兆（姿态角速度上升 / CoM 越界趋势），即"可预判"。

**RQ4（实时性）**：C++ 批量仲裁在 2048 环境、50 Hz 控制周期下的开销是多少？

> H4：单步总开销 < 1 ms，占控制周期 < 5%。

## 6. 实验设计

| 配置 | 说明 |
| --- | --- |
| **A（基线）** | 无仲裁（v0.1 行为） |
| **B（固定阈值）** | 阈值取基线分布的 99.5 分位（物理量：CoM 余量、摩擦锥余量、力矩余量） |
| **C（自适应阈值）** | 阈值随命令速度/地形难度缩放 |

| 场景 | 环境 | 目的 |
| --- | --- | --- |
| S1 平地 | `Go2Loco-S1-Flat-Blind-v0` | 基本有效性与性能损失 |
| S2 扰动 | `Go2Loco-S2-Disturbance-v0` | 干预是否集中于扰动时刻 |
| S2b 轻地形 | `Go2Loco-S2b-LightTerrain-v0` | 泛化到地形变化 |

- 评估种子：冻结集 `E={0..19}`，每配置 20 episodes；
- 策略：S1 三种子（P0 产出），A/B/C 共用同一策略权重，**只改仲裁层**；
- 指标：危险状态率（摔倒 + CoM 越界 + 摩擦锥违背）、成功率、干预率、触发分布、平均奖励、仲裁开销（µs/env/step）。

## 7. 交付物

1. **架构文档**（本文）+ 层间契约与仲裁状态机
2. **C++ 核心**：`cpp/guardrail/`（FK、CoM/支撑多边形、摩擦锥、RNID、批量裁决）+ Catch2 测试 + 基准
3. **绑定**：pybind11 批量接口（numpy ↔ Eigen）
4. **Python 编排**：`mora/arbitration.py`（契约 + AllowAllArbiter + 状态机）、`locomotion/guardrail.py`（Go2 绑定）
5. **模型导出**：`scripts/export_go2_model.py` + 一致性测试
6. **实验**：A/B/C × 3 场景 × 20 episodes + 触发归因
7. **报告**：`reports/guardrail/`，含证据束（run_config + SHA-256）
8. **演示**：同一 episode 在有/无仲裁下的对比视频（含触发时刻标注）
9. **README 首屏**：架构图 + 一句话结论 + 实时性数据

## 8. 时间盒（3 周 + 1 周包装）

| 周 | 内容 | 里程碑 |
| --- | --- | --- |
| W1 | P0 三种子训练（2.2 h）；Go2 参数导出；C++ FK/CoM 与支撑多边形 | 三种子证据束；C++ FK/CoM 与 Isaac 数值一致 |
| W2 | 摩擦锥 + RNID + 批量 pybind；Python 契约与状态机；基准测试 | 2048 envs 仲裁 < 1 ms；单元测试全绿 |
| W3 | Go2 集成、阈值标定、A/B/C × 3 场景、触发归因 | 结果表 + 触发分布 + 实时性数据 |
| W4 | 报告、README、视频、演练 | 可复现 + 可讲 |

**分期与砍单**：C++ 部分先做 FK/CoM/支撑多边形（必做），再做摩擦锥（必做），RNID 为加分项；若 RNID 来不及，用"PD 力矩限幅"替代，并在报告中说明。

## 9. 成功判据

**必须：**
- [ ] 契约实现且向后兼容（默认 AllowAllArbiter 不改变 v0.1 行为）
- [ ] C++ FK/CoM 与 Isaac 的数值对比误差达标（位置 < 1e-4 m）
- [ ] 支撑多边形与摩擦锥检查有边界单元测试
- [ ] 批量仲裁在 2048 envs 下 < 1 ms/step（有基准数据）
- [ ] A/B/C × 3 场景 × 20 episodes 完整
- [ ] 报告回答 RQ1–RQ4（成立或不成立并解释）
- [ ] 阈值标定可复现（分位数来源）
- [ ] 证据束 + 一键命令 + CI 绿

**加分：** RNID 力矩可行性；触发记录定位到具体失败模式并回馈训练分布。

## 10. 边界与风险

**不做**：新 RL 算法、真机、VLA、S3 楼梯、多机。

| 风险 | 预案 |
| --- | --- |
| C++ 动力学实现出错 | 先与 Isaac 的 FK/CoM 对比再接入；单元测试覆盖边界 |
| 2048 envs 批量接口性能不达标 | 批量化（一次处理所有 env）；若仍超预算，降采样仲裁（每 2 步检查一次）并报告 |
| 仲裁频繁触发导致成功率大跌 | 报告 Pareto 曲线；阈值上调至 99.9 分位；"性能损失"作为结果 |
| 阈值取自分位数对扰动场景过严 | C 配置按场景缩放；报告中给出两套阈值 |
| 仲裁掩盖策略缺陷 | 区分"被拦截的危险动作"与"策略本就不会做"；与 A 的失败样本对照 |
| 时间不足 | 保 FK/CoM + 支撑多边形 + 摩擦锥；RNID 降级为 PD 力矩限幅 |

## 11. 与 Phase-1 的关系

Phase-1 的 M2.1 证明了"事后物理停机"可行（失败后 StopSkill 降速）。v0.2 把安全机制前移为**事前仲裁**，并将它从"某个 adapter 的配置项"提升为"架构层契约"。两者构成同一条演进线：

```text
事后停机（M2.1） → 事前仲裁（v0.2，C++ 物理检查） → 形式化安全过滤器（未来，MPC barrier / CBF）
```
