# C1 baseline: go2-convex-mpc 接入位（占位）

对应冻结协议 [EXPERIMENT_DESIGN.md](../../../docs/EXPERIMENT_DESIGN.md) §6 的 **C1**：
外部仓库 [`elijah-waichong-chan/go2-convex-mpc`](https://github.com/elijah-waichong-chan/go2-convex-mpc)
（MuJoCo + Pinocchio 凸 MPC），接入同一评估循环，与 RL 策略**按种子配对**对比（RQ2）。

## 状态：待许可审查，未引入任何代码

按协议要求，接入前必须完成：

- [ ] **许可审查**：确认上游许可证允许本仓库的接入方式（封装调用 vs. 子模块 vs.
      vendor），结论与许可证文本登记到
      [`docs/THIRD_PARTY_NOTICES.md`](../../../docs/THIRD_PARTY_NOTICES.md)
- [ ] **接入方式决策**：优先封装调用（wrapper），保持上游代码不被修改，便于溯源
- [ ] **同一评估循环**：C1 跑在与其他条件相同的种子集 `E = {0..19}` 和
      `locomotion/eval/metrics.py` 指标上，逐种子配对

## 边界

在上述两项完成前，本目录不应出现任何来自上游仓库的代码或衍生实现。
