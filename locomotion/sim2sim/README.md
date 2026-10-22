# S4: Isaac → MuJoCo Sim2Sim（设计占位）

对应冻结协议 [EXPERIMENT_DESIGN.md](../../docs/EXPERIMENT_DESIGN.md) 的 **RQ4 / S4**：
在**零策略改动**的前提下，把 S1/S2 在 MuJoCo 中复刻，量化 Sim2Sim 差距
（成功率阈值 ≥ 90%），并归因主导因素（执行器模型 / 延迟 / 接触参数）。

## 冻结约束

- 与 Isaac 侧使用**同一命令序列、同一评估种子集** `E = {0..19}`（`locomotion/eval/seeds.py`）；
- 指标复用 `locomotion/eval/metrics.py` 的单一实现，禁止在 MuJoCo 侧另写一套口径；
- 每个被归因因素的切换（执行器模型、延迟、接触参数）单独记录为一次 run，
  按仓库证据约定生成 `run_config.json` / `events.jsonl` / SHA-256 清单。

## 待办（M2–M3 接入时完成）

- [ ] 引入 Unitree Go2 官方 MuJoCo 模型（`models/` 目前只有 Go2W，缺 Go2）；
      记录来源与许可到 `docs/THIRD_PARTY_NOTICES.md`
- [ ] 执行器建模决策：position-actuator 谐振频率/阻尼 vs. 官方 PD 增益，写明与
      Isaac 侧 `actuator` cfg 的对应关系
- [ ] 策略导出链路：Isaac checkpoint → ONNX/jit → MuJoCo 推理封装（零改动原则的
      操作定义：权重与观测归一化统计逐位一致）
- [ ] 命令序列生成器：按种子确定性地生成每个 episode 的 (vx, wz) 序列，两侧共用
- [ ] 归因实验矩阵：基线 S4 + 逐因素切换（执行器 / 延迟 / 摩擦），每格 20 episodes

## 明确边界

本目录在 M1 结束前不应有可执行代码；提前实现属于范围外变更。
