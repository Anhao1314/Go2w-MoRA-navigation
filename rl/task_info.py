"""面板/详情页展示用的任务说明、要求与目标（纯数据，零依赖）。"""

from __future__ import annotations

TASK_INFO: dict[str, dict] = {
    "balance": {
        "name": "Balance v2 · 抗撞击站立",
        "desc": (
            "机器狗以四轮站姿保持平衡，RL 只输出两个后轮力矩，腿保持四轮脚本姿态。"
            "每个 episode 随机安排 2~4 次沿 ±x 方向的 40~100N 力脉冲撞击"
            "（含等效俯仰力矩），目标是像被推搡一样在干扰中保持站立。"
        ),
        "requirements": [
            "10s episode 内经受 2~4 次随机撞击而不倒",
            "全程 |pitch| ≤ 0.35 rad",
            "基座高度 base_z ≥ 0.30 m",
            "无 NaN；评估 5 个 episode 全部通过",
        ],
        "goal": "抗撞击保持站立：max_dev ≤ 0.35、min_clear ≥ 0.30、成功率 100%。",
        "params": "观测 44 维 / 动作 2 维（后轮力矩 ±15Nm，100Hz）· 10s/episode · 8M 步（加速队列 4M），从零训练，可开课程学习。",
    },
    "full_chain": {
        "name": "Full Chain · 四轮→双轮→四轮动作链",
        "desc": (
            "从 balance 最优权重初始化。按脚本先收腿成双轮，双轮平衡保持至少 5s，"
            "再放腿恢复四轮并稳定站立；RL 只输出两个后轮力矩，腿阶段由脚本切换。"
        ),
        "requirements": [
            "双轮保持 ≥ 5s（偏差 ≤ 0.1 rad、前轮离地 ≥ 0.10 m）",
            "恢复四轮后 |pitch| ≤ 0.05 rad 稳定 ≥ 2s",
            "15s 内完成完整动作链；无 NaN",
        ],
        "goal": "完整动作链成功执行并平稳恢复四轮：dual_hold ≥ 5s、settle ≥ 2s、recovered=True。",
        "params": "观测 44 维 / 动作 2 维 · 15s/episode · 8M 步，用 balance best 初始化（--init-from）。",
    },
    "full_chain_simple": {
        "name": "Full Chain 简化版 · 双轮保持课程",
        "desc": (
            "full_chain 的中间课程：四轮站姿 → 收腿 → 双轮保持 5s 即成功终止，"
            "不要求放腿恢复四轮。观测/动作维度与 full_chain 一致，"
            "训练出的 best 再作为完整 full_chain 的初始化。"
        ),
        "requirements": [
            "双轮保持 ≥ 5s（偏差 ≤ 0.1 rad、前轮离地 ≥ 0.10 m）",
            "成功后终止并 +30；中途摔倒 -20",
            "12s episode；无 NaN",
        ],
        "goal": "稳定完成双轮保持课程：dual_hold ≥ 5s，作为 full_chain 的初始化权重。",
        "params": "观测 44 维 / 动作 2 维 · 12s/episode · 4M 步，可选用 balance best 初始化。",
    },
    "traverse_slope": {
        "name": "Traverse · 缓坡穿越（15~25°）",
        "desc": (
            "平地接 15~25° 缓坡，上坡穿越 4.5m 到达目标。地形为运行时生成的高度场"
            "（平滑坡面 + 随机凸起，逐 episode 随机化）。RL 输出 4 轮力矩 + 前后腿组"
            "目标姿态（共 6 维），可主动蹲跳/调姿。"
        ),
        "requirements": [
            "到达目标 x ≥ 4.5m（未到达目标算失败）",
            "巡航 ≤ 0.8 m/s（超速惩罚），尽量不回退、不偏航",
            "单场景成功率 ≥ 60%，整体 ≥ 80%；无 NaN",
        ],
        "goal": "稳定上坡并到达目标：成功率、平均穿越距离、最大俯仰偏差均进入验收。",
        "params": "观测 57 维 / 动作 6 维 · 100Hz · 8M 步（加速队列 4M）· 高度场地形（--terrain hfield）。",
    },
    "traverse_flat_slope": {
        "name": "Traverse · 陡坡穿越（25~35°）",
        "desc": (
            "平地接 25~35° 陡坡，上坡穿越 4.5m 到达目标。地形为运行时生成的高度场"
            "（平滑坡面 + 随机凸起），对爬坡动力和腿部调姿要求更高。"
        ),
        "requirements": [
            "到达目标 x ≥ 4.5m（未到达目标算失败）",
            "巡航 ≤ 0.8 m/s（超速惩罚），尽量不回退、不偏航",
            "单场景成功率 ≥ 60%，整体 ≥ 80%；无 NaN",
        ],
        "goal": "稳定爬上陡坡并到达目标：成功率、平均穿越距离、最大俯仰偏差均进入验收。",
        "params": "观测 57 维 / 动作 6 维 · 100Hz · 8M 步（加速队列 4M）· 高度场地形（--terrain hfield）。",
    },
    "traverse_curve": {
        "name": "Traverse · S 形弯道穿越",
        "desc": (
            "S 形弯道走廊（中心线 y=A·sin，走廊半宽 0.45m），按弧长计进度，"
            "需要差速转向沿中心线行驶。弯道保持盒状走廊结构（不升级高度场）。"
        ),
        "requirements": [
            "不越界：横向偏差 |dev| > 0.45m 立即终止",
            "按弧长到达终点（CURVE_TOTAL_LEN）",
            "单场景成功率 ≥ 60%，整体 ≥ 80%；无 NaN",
        ],
        "goal": "沿弯道中心线稳定通过：越界次数最少、成功率达到验收。",
        "params": "观测 57 维 / 动作 6 维 · 100Hz · 8M 步（加速队列 4M）· 盒状走廊。",
    },
}


def task_info(task: str) -> dict | None:
    """返回指定任务的信息字典；未知任务返回 None。"""
    return TASK_INFO.get(task)
