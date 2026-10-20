# 用户视频 → GVHMR → GMR → 元气仔

输入：`f5a1f8328da750a8f4a2c2e50c2c8393.mp4`，133 帧、30 FPS、1080×1920。动作约 4.4 秒，包含转身、抬臂和小幅跳动。

项目在训练主机 `robotit@100.101.111.80` 的 `/home/robotit/honor_ws/video_motion_pipeline`。本地保留脚本、结果和说明。所有视频推理均在这台主机进行。

## 此次结果

已完成整段 GVHMR 恢复、元气仔重定向和原速回放，输出 221 帧、50 Hz、4.4 秒。手臂方向相对 GVHMR 人体估计的平均偏差：左上臂 8.15°、左前臂 6.25°、右上臂 8.34°、右前臂 4.64°。脚底最低间隙约 1.9 mm，无明显穿地。平滑后有 54 个关节采样值需裁回模型限位，已记录在 manifest。

另外完成 600 次 PPO Tracking 微调（2048 环境、每轮 24 步，共 29,491,200 个仿真样本）。选用 `model_599.pt`，本地保存为 `artifacts/experimental_model_599.pt`。

| 测试 | 完成情况 | 关节 RMSE | 相对身体位置 RMSE | 根节点位置 RMSE |
| --- | --- | --- | --- | --- |
| 原官方 Tracking，固定初态 | 0/16，平均约 1.74 秒失稳 | 0.329 rad | 0.201 m | 0.642 m |
| 视频微调后，固定初态 | 16/16 完成 4.36 秒 | 0.242 rad | 0.070 m | 0.261 m |
| 视频微调后，小扰动，seed 123 | 32/32 完成 4.36 秒 | 0.243 rad | 0.070 m | 0.277 m |

固定初态的 16 个环境重复相同条件，不等于 16 个独立场景；小扰动覆盖轻微位置、姿态、速度、关节初值和观测噪声。所有测试均未越过参考末尾循环，未使用虚拟支撑。动态片段比运动学参考短 0.04 秒，来自官方格式转换与末尾防循环留帧。

**当前可完整执行，但跟踪精度尚未通过验收。** 预先采用的门槛为完成率 ≥95%、关节 RMSE ≤0.15 rad、相对身体位置 RMSE ≤0.06 m、根节点位置 RMSE ≤0.20 m、无意外循环。完成率达标，后三项误差仍超标。训练中间的 100/200/400 次检查点也保留了失败记录，不能用训练平均回合长度代替全片测试结论。

主要待处理项是开头转身的接触可行性、肩部 IK 分支连续性和局部过快切换。当前参考峰值关节速度约 31.4 rad/s，位于肩俯仰；它是运动学轨迹速度，不代表电机已能精确执行。视频里的手指、表情不在 29 关节映射范围内；单目三维恢复也没有真实三维动捕作为精度基准。

查看入口：

- 首先看 `outputs/vita_boy/video_to_vita_comparison.mp4`，确认动作内容。
- 再看 `outputs/tracking_attempt.mp4`，这是实际物理仿真执行，包含上述跟踪偏差。
- [本地网页实验回放](http://127.0.0.1:8090)：已替换原跑步网页，点击“播放 / 重新播放”。页面明确标注未通过精度验收，片段末尾自动暂停。
- 测试原始记录：`outputs/eval_599.json`、`outputs/eval_599_jitter.json`；对应逐帧轨迹为同名 `.trajectory.json`。

相机固定的假设已用背景特征检查：在 540×960 推理尺寸下，抽查 4 个时刻的背景中位位移均小于 0.04 像素。腕、踝关键点的平均热图分数约 0.91，未出现低于 0.3 的帧；该分数是模型置信度，不是三维准确率。

## 链路和环境

1. 保存原视频及 SHA256；另生成 540×960、相同帧数和时间的推理副本。
2. 使用 [GVHMR 官方代码](https://github.com/zju3dv/GVHMR)，固定提交 `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`。按固定相机处理；YOLO 跟踪、ViTPose 关键点、HMR2 特征、GVHMR 全局 SMPL-X 恢复。
3. 使用 [GMR 固定版本](https://github.com/YanjieZe/GMR/tree/bb1bbe40774794fceb2a7c579a3464a28e68c844) 的 SMPL-X 世界坐标转换和 Mink 两阶段 IK，重定向到工作空间中实际的 VitaBoy MJCF。
4. 将人体和机器人在相同 T 姿态下校准各连杆坐标，处理元气仔前臂沿局部 +X 轴的结构。机器人有 29 个关节，关节顺序写在 NPZ 和 manifest 中。
5. 手臂以肘、腕位置约束为主，配合弱姿态正则和多个初始解，避免抬臂后的局部 IK 卡死。保留原始时间，短窗平滑并重采样到 50 Hz；在推测的支撑阶段修正脚底与地面的高度偏移。

GVHMR 使用独立 Python 3.10 环境 `.venv`，PyTorch 2.3.0+cu121；GMR IK 和渲染使用现有 mjlab 环境。没有改动原行走、跑步策略或官方仓库源码。SciPy 需 ≥1.14 才支持 GMR 所用的 `scalar_first` 四元数 API。

人体模型和推理权重采用 [GVHMR 官方演示的资源清单](https://huggingface.co/spaces/LittleFrog/GVHMR/blob/main/app.py)，来源、大小和 SHA256 保存在 `assets`。大型 CUDA wheel 使用 NVIDIA 官方源并校验 PyPI 记录的 SHA256。模型文件保留在主机，不随结果重复复制。

## 重建与重定向

在远程主机运行：

```bash
cd /home/robotit/honor_ws/video_motion_pipeline
bash run_reconstruction.sh
../honor_rl_mjlab/.venv/bin/python check_motion_quality.py
```

已有 GVHMR 结果会复用。换视频时应使用新的输出目录，避免复用另一段视频的预处理缓存。

主要结果：

- `outputs/user_video/1_incam.mp4`：人体网格叠加到视频。
- `outputs/user_video/2_global.mp4`：人体三维重建。
- `outputs/user_video/hmr4d_results.pt`：原始 GVHMR 输出。
- `outputs/human_skeleton.npz`：世界 Z 向上、带名称和父子层级的人体骨架。
- `outputs/vita_boy/video_to_vita_comparison.mp4`：原视频、重建人体、元气仔三栏同步回放。
- `outputs/vita_boy/retargeted_qpos.npz`：50 Hz 元气仔自由基座和 29 关节参考；基座四元数 **wxyz**。
- `outputs/vita_boy/video_motion.csv`：官方转换器输入；列为 xyz、基座四元数 **xyzw**、29 关节。
- `outputs/vita_boy/smplx_to_vita_boy.json`：专用 GMR 配置。
- `outputs/vita_boy/quality_report.json`、`motion_quality.png`：运动学检查；误差相对重建人体计算，不代表与真实人体三维姿态的误差。

## Tracking 动力学验证

先转换成官方 Tracking 的参考格式：

```bash
cd /home/robotit/honor_ws/honor_rl_mjlab
.venv/bin/python datasets/csv_to_npz.py \
  --input-file ../video_motion_pipeline/outputs/vita_boy/video_motion.csv \
  --output-name /home/robotit/honor_ws/video_motion_pipeline/outputs/vita_boy/video_motion.npz \
  --input-fps 50 --output-fps 50
```

`train_tracking_video.py` 为该视频建立单独实验 `vita_boy_video_tracking`，从官方 Tracking actor 初始化，默认 2048 个环境。`evaluate_video_tracking.py` 复用之前验证过的有限片段评估，不跨参考动作末尾循环；检查完成率、关节误差、相对身体误差、根节点误差及意外跳帧。该动作不是跑步片段，因此不采用跑步速度或腾空次数作为通过条件。

```bash
cd /home/robotit/honor_ws/honor_rl_mjlab
.venv/bin/python ../video_motion_pipeline/train_tracking_video.py --iterations 600
.venv/bin/python ../video_motion_pipeline/evaluate_video_tracking.py \
  --checkpoint /absolute/path/to/model_XXX.pt --num-envs 32 --jitter \
  --output /home/robotit/honor_ws/video_motion_pipeline/outputs/tracking_eval.json
```

恢复此次实验网页：

```bash
cd /home/robotit/honor_ws/honor_rl_mjlab
.venv/bin/python ../video_motion_pipeline/play_video_tracking.py \
  --checkpoint ../video_motion_pipeline/artifacts/experimental_model_599.pt --port 8090
```

实际 GPU 训练目录记录于 `outputs/training_run.txt`；`artifacts/params` 保存对应环境和训练参数。项目源代码、输出和资源清单的 SHA256 记录于 `sha256_manifest.json`，大模型权重另见 `assets/sha256_models.json`。

运动学重定向与动力学执行是不同结果：直接设置 qpos 得到的对照视频只能展示动作参考，动力学结论以实际 Tracking 评估记录为准。本项目范围是仿真，未向实机下发动作。
