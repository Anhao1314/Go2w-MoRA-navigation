# Third-Party Notices

## westonrobot/unitree_mujoco (BSD 3-Clause License)

本项目 `rl/terrain.py` 的运行时高度场地形生成器，其“Perlin 噪声生成高度场”
思路参考自：

- 项目：https://github.com/westonrobot/unitree_mujoco
- 文件：`terrain_tool/terrain_generator.py`（`AddPerlinHeighField`）

### 适配说明

原实现使用 `noise` + `cv2` 生成 PNG 高度图，再通过 XML 的 `<hfield file=...>`
加载。本项目改为：在 MuJoCo 3.11 运行时用 numpy 生成归一化高度矩阵，直接写入
`model.hfield_data`（逐 episode 随机化，无需新增依赖、无需重新编译模型）。
仅借鉴“Perlin 高度场”算法思路，未复制原代码。

### BSD 3-Clause License

Copyright (c) 2025, Weston Robot
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
