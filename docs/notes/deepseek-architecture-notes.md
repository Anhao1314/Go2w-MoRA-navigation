# DeepSeek 2026 架构开源调研

[返回首页](../README.md) · [文档索引](README.md) · [研究问题](RESEARCH.md) · [架构 v0.2](ARCHITECTURE_V2.md)

> 调研日期：2026-10-23。原则：只记录在 GitHub 原始页面上核实过的内容；搜索结果里未经核实的数字（如 V4 的参数量）单独标注。

## 1. 核实结果

| 项目 | 位置 | 核实状态 | 内容 |
| --- | --- | --- | --- |
| **Engram** | [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) | ✅ 真实，Apache-2.0，约 4.8k stars | 论文 PDF + 架构图 + `engram_demo_v1.py`（demo 里 Attention/MoE/mHC 均为 mock）。**没有完整训练代码，也没有放出模型** |
| **mHC**（流形约束超连接） | 论文 [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) | ✅ 论文存在；**未找到 DeepSeek 官方仓库** | 社区实现：[tokenbender/mHC](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)、Chenhao-Guan/deepseek-mhc、AndreSlavescu 的 CUDA 版 |
| **DeepInfer** | [github.com/deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) | ✅ 真实，2026-10-21 更新 | 推理框架（基于 vLLM），不是架构研究 |
| **DeepSeek-V4 架构细节** | 未在 GitHub 组织页核实 | ⚠️ 未确认 | 搜索摘要称 2026-04 发布、1.6T MoE、混合注意力 CSA/HCA；组织页只显示 45 个仓库中的 10 个，未见 V4 仓库。**这些数字不引用** |

组织页（deepseek-ai）最近更新的仓库：`DeepInfer`、`DeepEP`、`DeepGEMM`、`deepseek-recipe`、`DeepJIT`、`deepseek-harness`、`FlashMLA`、`DeepGEMM-Ascend`、`TileKernels`。

## 2. Engram：条件记忆（"查"与"算"分离）

- **主张**：除了 MoE 的"稀疏计算"，还存在第二条稀疏轴——**条件记忆**。把经典 N-gram 嵌入现代化为 O(1) 的确定性查表，检索静态记忆并与动态隐状态融合。
- **可 offload**：因为寻址是确定性的，巨大的嵌入表可以放到 host memory，推理开销很小。
- **U 型缩放律**：在"神经计算（MoE）"与"静态记忆（Engram）"之间存在最优容量分配。README 没有给出具体比例（细节在论文里）。
- **结果**：等参数、等 FLOPs 下，Engram-27B 在知识、推理、代码、数学上优于 MoE 基线。
- **机制解释**：Engram 让浅层不必再"重建静态模式"，把有效深度留给复杂推理。

## 3. mHC：用约束代替自由学习

- **做法**：把残差连接的混合矩阵约束到 **Birkhoff 多面体**（双随机矩阵），用 Sinkhorn-Knopp 投影实现；`H_pre`/`H_post` 用 softmax 保持非负。
- **目的**：保证信号在深层网络中不放大、不衰减，同时保留表达能力。
- **实现**：社区版有 nanoGPT 集成（FineWeb10B）、ResNet 集成、以及 CUDA 加速版。

## 4. 对本项目的启发

### 4.1 "查 vs 算"直接对应 MoRA 的分层

| Engram | 本项目 |
| --- | --- |
| 静态记忆查表（确定性、O(1)） | S2 规则规划（阶段管理、分支锁定） |
| 神经计算（MoE） | S1 策略（学习） |
| U 型律：记忆与计算存在最优分配 | **开放问题**：分层架构中"规则层"与"学习层"的容量如何分配？ |

这给研究主线补了一个理论参照：分层不只是"拆开更简单"，还可以理解为**在不同层之间分配"确定性查表"与"学习计算"**。若时间允许，可以作为讨论章节或未来工作。

### 4.2 mHC 的约束思想可以迁移到"层间契约"

mHC 用双随机约束保证信号传播稳定；本项目的层间约束（仲裁层、动作裁剪、支撑多边形检查）是同一类思想：**用结构化约束替代纯学习**。报告里可以引用它作为"约束式架构"的理论支撑，说明这不是孤立的工程技巧。

### 4.3 Engram 的 demo 写法值得借鉴

它把 Attention/MoE/mHC 全部 mock 掉，只保留 Engram 数据流——这正是"架构验证"应有的最小实现：先证明数据流与接口，再谈性能。这和我们 `mora/arbitration.py` 里 `AllowAllArbiter` 的思路一致。

## 5. 边界

- 这些都是 **LLM 架构**，不是机器人/具身架构。不能直接照搬，只能借思想。
- Engram 没有放出训练代码和模型，无法复现其结论；引用时只能作为"思想来源"。
- V4 的具体架构数字未经核实，不写进任何结论。

## 6. 建议

1. 在 [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) 的"相关工作"里加一小段，引用 Engram（稀疏轴）与 mHC（约束式连接），说明本项目的层间约束属于同一思想谱系。
2. 不改变当前 A/B 实验设计；"规则层/学习层容量分配"作为未来假设记录在 [RESEARCH.md](RESEARCH.md) 的扩展部分。
3. 若后续要验证 4.2 的思想，可以在仲裁层里加"约束强度"维度（如限制动作变化率），而不只是硬阈值。
