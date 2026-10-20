# 太极拳视频 → 元气仔动作重定向

输入视频：`fbf6fdff4d44dc198904dcb988964fa0.mp4`，1280×580、30 FPS、1284 帧，约 42.8 秒。处理整段视频，保持原始动作时间，没有加速播放。

远端目录：`robotit@100.101.111.80:/home/robotit/honor_ws/taichi_motion_pipeline`。本地目录保留脚本、动作文件、检查结果和视频。

## 查看结果

- `outputs/vita_boy_collision_free/taichi_to_vita_comparison.mp4`：自碰撞修正后的三栏同步对比。
- `outputs/vita_boy_collision_free/taichi_vita_kinematic.mp4`：修正后的机器人单独回放。
- `outputs/vita_boy_collision_free/self_collision_before_after.jpg`：27.80、34.24、42.52 秒同视角修复前后对比。
- `outputs/vita_boy_collision_free/comparison_frames.jpg`：八个时刻的动作检查图。
- `outputs/user_video/1_incam.mp4`：GVHMR 人体网格叠加到原视频，保留原始画幅。
- `outputs/vita_boy_collision_free/motion_quality.png`、`quality_report.json`：修正后的重定向质量检查。

原 `outputs/vita_boy` 完整保留，含未约束自碰撞的旧参考。后续 Tracking 使用 `outputs/vita_boy_collision_free/taichi_motion.npz`。

**这些是运动学参考回放。** 本次没有训练太极拳 Tracking 策略，没有进行自由动力学稳定性验证，也没有向实机下发动作。回放中直接设置机器人姿态，不能据此断言机器人能在重力和接触作用下完成全部动作。

## 原版本检查结果（自碰撞修复前）

输出参考共 2139 帧、50 Hz，首末帧时间跨度 42.76 秒。对应官方 Tracking NPZ 为 2138 帧，来自转换器的末端采样处理。

| 项目 | 结果 |
| --- | --- |
| 左上臂／左前臂平均方向偏差 | 6.78°／5.09° |
| 右上臂／右前臂平均方向偏差 | 10.82°／3.72° |
| 最低脚底间隙 | 1 mm，已消除穿地 |
| 基座高度范围 | 约 0.608–0.819 m |
| 关节速度绝对值第 99 百分位 | 约 1.48 rad/s |
| 最大关节速度 | 约 7.03 rad/s，仍需在后续动力学训练前关注局部快速变化 |

平滑后 195 个采样值略超模型限位，最大约 0.0104 rad；已按模型限位及微小余量裁剪。包含安全余量处理在内，共调整 1232 个关节采样值；最终参考均在限位内。这些是运动学数据检查，不是电机执行能力或碰撞安全认证。

右腕约 1.25% 的帧热图分数低于 0.3，其余三个腕／踝检查点未出现低于 0.3 的帧；遮挡片段仍可能产生姿态估计偏差。方向误差及图表均相对恢复的人体估计计算。

## 动作文件

- `outputs/human_skeleton.npz`：原始 GVHMR 导出的人体骨架，世界 Z 轴向上，包含关节名称、父子关系和姿态。
- `outputs/vita_boy/raw_gmr_qpos.npz`：GMR 原始 IK 输出，30 Hz。
- `outputs/vita_boy/retargeted_qpos.npz`：50 Hz 元气仔参考，包含自由基座 xyz、基座四元数 **wxyz**、29 个关节位置，以及脚底间隙。
- `outputs/vita_boy/video_motion.csv`：列顺序为 xyz、基座四元数 **xyzw**、29 个关节，供官方格式转换器使用。
- `outputs/vita_boy/taichi_motion.npz`：已转换为官方 Tracking 使用的关节及身体状态格式；“格式可用”不代表已训练策略。
- `outputs/vita_boy/smplx_to_vita_boy.json`：本次元气仔 GMR 映射。
- `outputs/vita_boy/manifest.json`：帧数、时间、关节顺序、处理参数和几何检查数值。

## 本次处理

复用此前安装好的 [GVHMR](https://github.com/zju3dv/GVHMR) 与 [GMR](https://github.com/YanjieZe/GMR/tree/bb1bbe40774794fceb2a7c579a3464a28e68c844) 环境和模型文件。GVHMR 在完整原始画幅上处理；对比视频中的人物裁剪只用于展示，不改变推理输入。每段视频使用独立输出目录，避免复用上一段视频的缓存。

GVHMR 恢复 SMPL-X 动作后，按实际 shape 参数维度分批计算人体骨架，避免长视频一次性求值占用过多内存。GMR 使用工作空间实际 VitaBoy MJCF、29 关节限位以及经过 T 姿态校准的连杆坐标。手臂主要跟踪肘和腕的位置，使用多个初始解，并提高连续性代价以减少慢动作中的 IK 分支跳变。

手臂采用 15 个源帧的居中平滑窗口，其他关节采用 7 帧窗口，之后重采样到 50 Hz。窗口按原始时间居中，不改变视频播放速度。

视频内的太极拳动作始终至少有一只脚支撑。针对单目恢复的地面高度漂移，按实际机器人脚底碰撞体调整基座 Z，使较低的一只脚靠近地面，并修正剩余穿地。另一只脚的相对抬升、各关节姿态及时间不因这一步改变。对比视频的人体骨架也做了地面显示对齐；原始人体骨架另存，未经该显示修正。地面先验只适用于这段无腾空视频，不能直接复用到跑步或跳跃动作。

手指和掌型没有映射到这套 29 关节模型。遮挡时的腕部姿态、单目估计的深度和绝对位移仍有不确定性；质量报告中的方向误差相对 GVHMR 人体估计计算，不是真实三维动捕误差。

## 手臂自碰撞修复

完整修复已通过验收，结果见 `outputs/vita_boy_collision_free/validation.json` 和 `self_collision_report.json`：

| 项目 | 最终结果 |
| --- | --- |
| 采样与范围 | 200 Hz、8553 点、89 对上肢碰撞几何 |
| 穿模采样点 | 0 |
| 最小几何间距 | 7.918 mm |
| 原版最深穿模反例 | 27.80 秒双手，约 72.02 mm 穿入 |
| 修正后的平均方向偏差：左上臂／前臂 | 7.11°／5.71° |
| 修正后的平均方向偏差：右上臂／前臂 | 10.52°／4.57° |
| 关节限位、有限值、单位四元数、脚底检查 | 通过，最低脚底间隙 1 mm |
| 腿、腰、基座与原参考逐值比较 | 完全一致 |
| 手臂帧间角速度上限 | 2.5 rad/s |
| CSV 转出的 Tracking 关节位置与参考一致性 | 通过 |

旧版只约束了关节限位、人体目标和脚底，缺少手臂与身体之间的距离约束。直接设置 qpos 的运动学回放不会受物理接触力纠正。单目遮挡、机器人与人体比例差异、腕部姿态歧义及平滑插值都会使双手交叉位置出现穿模，不能指望后续 RL 自动修好错误参考。

`repair_self_collision.py` 在所有平滑、50 Hz 重采样及脚底修正之后，进行逐帧的手臂约束优化。只调整 14 个肩、肘、腕关节，冻结自由基座、腰及腿，保持原视频时间轴；原姿态跟踪、修正量连续性、关节限位和每帧角度变化上限共同约束输出。逐帧目标间隙为 12 mm，手臂帧间变化上限为 2.5 rad/s；这不是电机能力认证。

之后对插值路径作加密检测，必要时由 `repair_interpolation.py` 联合调整发生问题区间的两个端点，同时约束前后相邻区间，在 200 Hz 采样点保留至少 8 mm 的局部修正目标。最后全段验收门槛为至少 5 mm，任何不合格区间都会使程序报错停止。不能只检查每帧端点后就宣称插值过程无碰撞。

检查使用实际 MJCF 编译后的碰撞网格凸包和世界坐标变换，由 [Python-FCL](https://github.com/BerkeleyAutomation/python-fcl) 计算有符号距离。当前 MuJoCo 3.10 对部分相交网格返回零距离和不稳定最近点，因此修正器使用独立 FCL 后端，避免在优化中混用不连续的距离结果。这里只改变项目内的几何检测依赖，不修改训练环境或机器人资产。

检查范围为 89 对几何：肩 yaw 以下的手臂与躯干、头颈、骨盆，以及两条手臂的所有相互配对。连接肩部的 pitch/roll 安装件不与躯干做距离约束，避免把关节装配处的 CAD 重合误当成动作碰撞。未覆盖所有腿部和同侧相邻连杆组合。验证将帧间关节线性插值加密到 200 Hz；这是离散加密检查，不是连续碰撞的数学证明。报告逐对记录最小间距和发生时刻。

几何修正仍属于运动学参考清洗。后续 Tracking 必须在真实碰撞配置下检查自由动力学跟踪、扭矩和接触，加入非期望身体接触惩罚及必要的提前终止，再评估实机部署。

已有 GVHMR/GMR 输出时，只重做自碰撞修复和导出即可，无需重新估计人体：

```bash
cd /home/robotit/honor_ws/taichi_motion_pipeline
PY=../honor_rl_mjlab/.venv/bin/python
OPENBLAS_NUM_THREADS=1 "$PY" repair_self_collision.py
"$PY" check_motion_quality.py --folder outputs/vita_boy_collision_free
MUJOCO_GL=egl "$PY" render_comparison.py --folder outputs/vita_boy_collision_free
MUJOCO_GL=egl "$PY" render_collision_review.py
cd ../honor_rl_mjlab
.venv/bin/python datasets/csv_to_npz.py \
  --input-file ../taichi_motion_pipeline/outputs/vita_boy_collision_free/video_motion.csv \
  --output-name ../taichi_motion_pipeline/outputs/vita_boy_collision_free/taichi_motion.npz \
  --input-fps 50 --output-fps 50
cd ../taichi_motion_pipeline
"$PY" validate_artifacts.py
```

独立依赖版本见 `collision_requirements.txt`。在当前主机安装到项目目录的命令为：

```bash
/home/robotit/.local/bin/uv pip install --python ../honor_rl_mjlab/.venv/bin/python \
  --target collision_vendor --no-deps -r collision_requirements.txt
```

## 复现

在已经配置好的训练主机执行：

```bash
cd /home/robotit/honor_ws/taichi_motion_pipeline
bash run_pipeline.sh
```

GVHMR 使用 `../video_motion_pipeline/.venv/bin/python`；IK、MuJoCo 渲染和格式转换使用 `../honor_rl_mjlab/.venv/bin/python`。已有重建结果会复用，再生成骨架、动作及视频。

输入视频 SHA256、置信度检查和显示裁剪范围见 `inspection`。版本和模型来源记录见 `assets`；交付文件校验表为 `sha256_manifest.json`。此前的跑步模型和其他视频项目不受本项目影响。
