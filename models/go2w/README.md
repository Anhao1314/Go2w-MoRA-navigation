# Unitree Go2w 轮腿机器人模型

本项目使用的机器人模型来自宇树科技官方开源仓库
[unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)，
采用 BSD-3-Clause 许可证（见本目录 `LICENSE`）。

- `go2w.xml`：官方 Go2w 轮腿机器人 MJCF 模型（未修改）。
- `assets/`：官方 3D 网格资源。
- `go2w_factory_scene.xml`：本项目的智慧工厂场景（无障碍平整地面 + 跟随相机），
  通过 `<include>` 引用官方模型。

轮式站姿关节角（髋 0 / 大腿 0.67 / 小腿 -1.3）来自宇树官方 SDK 示例
`example/go2w/low_level/go2w_stand_example.py`。
