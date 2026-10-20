# 元气仔跑步动作重定向与 Tracking 验证

完成日期：2026-10-19。工作主机：`robotit@100.101.111.80`。

## 本次完成的结果

已选取真实人体跑步动作，为元气仔建立 GMR 重定向配置，使用官方 Tracking 任务训练，并完成独立动力学及导出部署测试。

| 测试 | 完成情况 | 实际前向均速 | 关节角 RMSE | 相对身体位置 RMSE | 根部轨迹 RMSE |
|---|---:|---:|---:|---:|---:|
| mjlab 干净平地 | 1/1 完成完整片段 | 1.833 m/s | 0.113 rad | 3.36 cm | 9.64 cm |
| 小幅初始扰动＋观测噪声 | 96/96 完成完整片段 | 详见各评估 JSON | 全部通过判据 | 全部通过判据 | 全部通过判据 |
| 官方 Python TrackingState＋CPU MuJoCo | 完成完整片段 | 1.806 m/s | 0.113 rad | 3.56 cm | 8.95 cm |

96 次是三个评估随机种子各 32 个扰动试次，**不是 96 次独立训练**。本次训练种子为 42。独立评估种子为 123、456、789。

评估覆盖 196 个 50 Hz 控制步，即 **3.92 秒**。每次从动作参考的姿态及速度初始化，之后通过策略和物理仿真推进；没有虚拟吊带、悬浮支撑或参考轨迹传送。评估在动作结束前停止，`reference_wraps=0`。

本次结果证明的是选定短片段的跟踪能力。它不代表从静止起跑、长时间连续跑、复杂地形、AMP 算法、2 m/s 最终验收或实机部署已经完成。

## 查看结果

- [人体原动作与机器人重定向对照](artifacts/motion/human_to_vita_retarget.mp4)：两侧均是运动学参考，不是动力学证明。
- [参考动作与策略动力学对照](artifacts/reference_vs_tracking.mp4)：左侧为目标，右侧为策略在物理仿真中的实际运动。
- [独立 CPU 部署回放](artifacts/tracking_cpu_deploy.mp4)。
- [定量验证图](artifacts/tracking_validation.png)。
- [完整验证摘要](artifacts/validation_summary.json)。
- [导出的 Tracking ONNX](artifacts/tracking_run.onnx) 与 [配套动作 NPZ](artifacts/motion/run.npz)。
- [可继续训练的检查点](artifacts/model_500.pt)。

远端的实时回放已接入原来的 SSH 转发入口：`http://127.0.0.1:8090`。点击 **播放 / 重新播放** 开始，片段末尾自动暂停。重新打开已有页面后请刷新。该入口需要本机的 SSH 转发进程继续运行。

## 选择的动作

- 数据集：[Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)。
- 文件：`run2_subject4.bvh`。
- 范围：零起始帧号 `[6940, 7060)`，即 6940 至 7059，共 120 帧，原始帧率 30 Hz。
- 原人体片段净位移速度约 2.167 m/s；方向一致性约 0.9985。
- 按机器人尺寸重定向后，目标净位移速度约 1.820 m/s。
- 保持原时间尺度，`playback_time_scale=1.0`。没有通过加速播放提高速度。
- CSV 重采样为 50 Hz；官方转换器输出 198 帧 NPZ。有限片段评估保留末尾余量，防止命令重采样重置机器人状态。

原 BVH 的 SHA-256 为：

```text
cecbb82267dae06e4aabfaa761f7126b8bf384b38c159d92f9677b9c687100e7
```

数据来源、原许可和引用信息保存在 `artifacts/provenance/`。源数据许可为 CC BY-NC-ND 4.0，具体条款以保存的原文为准。

## 元气仔适配方式

GMR 固定版本：`bb1bbe40774794fceb2a7c579a3464a28e68c844`。训练仓库固定版本：`10c2ba02c7c001bb574771855b9dc41ef193e01d`。

使用训练仓库自身的元气仔 MJCF 和 29 个可控关节。IK 通过 GMR 的两阶段 FrameTask 与 Mink/DAQP 求解，按元气仔关节限位约束。

元气仔与 GMR 示例机器人的零位几何不同，尤其是前臂方向，不能直接复制其姿态偏置。本适配通过人体髋线、上下腿、脚尖、上臂及前臂的空间方向建立目标坐标系，再映射到元气仔连杆轴。骨盆目标保持竖直，躯干目标保留由人体脊柱方向得到的倾斜。

位置缩放采用腿部/根部 0.84、手臂 0.75。脚底几何由机器人实际碰撞体计算；根据源动作支撑阶段清理高度，飞行阶段插值修正，避免每帧将脚钉在地面。短窗滤波后重采样，重新计算速度。最终目标中脚底最深穿地约 2.4 mm，关节限位裁剪仅涉及一个采样值；运动学腾空比例约 13.1%。这些几何指标不替代物理测试。

完整映射见 `artifacts/motion/bvh_lafan1_to_vita_boy.json`，数据约定、关节顺序和质量数值见同目录 `manifest.json`。

## Tracking 训练

使用官方 `VitaBoy-Tracking` 的 PPO 任务，保留原执行器、逐关节动作缩放、154 维 actor 输入和 50 Hz 控制频率。基础训练阶段使用平地、小幅初始化扰动与观测噪声，暂未加入质量/质心随机化及外部推扰。

Actor 从官方 `deploy/checkpoints/tracking.onnx` 导入 MLP 和观测归一化参数，Critic、优化器重新初始化。导入数值检查保存在 `warm_start_validation.json`。最终导出的策略在真实回放观测上与 PyTorch 的最大动作绝对差约 `1.67e-6`（完整 FP32）。

训练使用 2048 个并行环境，每轮每环境采集 24 步，初始动作标准差 0.25、学习率 1e-4。任务原计划 2000 次迭代；第 500 号检查点已通过判据，因此在迭代 821 后结束了本次训练，保留远端至 750 号的检查点，**发布的是通过验证的 `model_500.pt`**。

对照结果：零动作策略在约 0.42 秒失败，直接参考关节位置 PD 在约 0.64 秒失败。官方原 Tracking 策略虽然能完成片段，但平均速度约 0.997 m/s、根部轨迹 RMSE 约 1.63 m，尚未跟上新跑步动作。训练后的策略显著缩小了这些误差。对照文件也保存在 `artifacts/`。

## 本阶段工程判据

以下是本次短片段验证的工程阈值，不是课程最终评分标准：

- 完整片段完成率至少 95%。
- 29 关节整体 RMSE 不超过 0.15 rad。
- 14 个选定身体连杆的、去根平移后的相对位置 RMSE 不超过 0.06 m。
- 根部全局轨迹 RMSE 不超过 0.20 m。
- 前向速度与参考平面速度的平均绝对差不超过 0.25 m/s。
- 无参考回绕或传送；无虚拟支撑。
- 排除前 0.5 秒后，真实足接触传感器测得的腾空比例至少 1%，并有至少两段独立腾空。

96 个扰动试次使用：初始 x/y ±1 cm，roll/pitch/yaw ±0.02 rad，关节角 ±0.02 rad，根速度各分量 ±0.05，并保留任务的观测噪声。这是温和扰动验证，不是地形或大推力鲁棒性验证。误差按失败前有效采样计算，因此必须和完成率一起阅读，不能将提前倒下后的少量低误差当作成功。

## 在远端复现

项目位置：`~/honor_ws/motion_pipeline`；训练框架：`~/honor_ws/honor_rl_mjlab`。

已有环境上重新生成数据：

```bash
cd ~/honor_ws/motion_pipeline
python3 fetch_gmr_source.py
python3 fetch_lafan_runs.py
~/.local/bin/uv pip install --python ../honor_rl_mjlab/.venv/bin/python \
  --target vendor --no-deps -r requirements_retarget.txt
../honor_rl_mjlab/.venv/bin/python retarget_run.py
cd ../honor_rl_mjlab
.venv/bin/python datasets/csv_to_npz.py \
  --input-file ../motion_pipeline/data/retargeted/lafan_run2_s4_geometry/run.csv \
  --output-name /home/robotit/honor_ws/motion_pipeline/data/retargeted/lafan_run2_s4_geometry/run.npz \
  --input-fps 50 --output-fps 50
```

评估选定检查点：

```bash
cd ~/honor_ws/honor_rl_mjlab
.venv/bin/python ../motion_pipeline/evaluate_tracking.py \
  --checkpoint ../motion_pipeline/artifacts/model_500.pt \
  --num-envs 32 --jitter --seed 123 --output /tmp/run_tracking_eval.json
```

独立 CPU 部署评估：

```bash
.venv/bin/python ../motion_pipeline/evaluate_tracking_deploy.py \
  --policy ../motion_pipeline/artifacts/tracking_run.onnx \
  --output /tmp/run_tracking_cpu.json
```

继续同一数据和配置的训练：

```bash
.venv/bin/python ../motion_pipeline/train_tracking_run.py \
  --resume /home/robotit/honor_ws/honor_rl_mjlab/logs/rsl_rl/vita_boy_run_tracking/2026-10-19_20-35-12_lafan_run2_s4_seed42/model_500.pt \
  --num-envs 2048 --iterations 1000
```

`--iterations` 是追加迭代数。评估脚本还支持 `--video`、`--export-onnx` 和 `--verify-onnx`。本项目脚本均针对开发主机仿真，不包含实机控制入口。

## 后续工作

本片段约 1.82 m/s 的参考速度尚未覆盖最终“均速不低于 2 m/s”验收。后续应补充更高速度及起停、转弯参考，接入真正的 AMP 专家采样、判别器和风格奖励，再训练速度可控策略。长片段连续性、地形课程、能耗、热约束和实机部署分别验证。当前结果是这一流程的动作重定向与基础 Tracking 里程碑。
