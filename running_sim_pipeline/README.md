# 元气仔指令驱动跑步：仿真训练与部署验证

## 消融实验完成（2026-10-26）

PPO、PPO＋动态LS-AMP、PPO＋域随机化、完整组合四组，各3个独立训练种子，共12次训练、19200次正式更新，已完成统一评估。正式学习耗时累计166.10分钟，含预检和测试的整轮墙钟耗时约3.49小时。
固定1600次更新预算下，四组地形完成率均值分别为42.27%、50.35%、90.80%、86.37%；种子差异较大，不能断言AMP稳定提高地形完成率。停止与地形检查仍有失败，未替换默认模型。
完整方法、逐种子结果和配对区间见[消融实验报告](ABLATION_REPORT.md)，复现见[消融代码说明](ablation/README.md)。此实验不能直接分解历史5020次更新交付模型的成因。

## 专项优化记录（2026-10-25）

本轮追加1400次正式更新，测试了起停课程、速度/停稳奖励与训练期直跑保留约束。起停转向得到改善，但未通过全部保留检查；默认交付模型保持原版。
结果与失败记录见 [专项优化报告](TRANSITION_OPTIMIZATION_REPORT.md) 和 [最终验证JSON](experiments/transition_20261025_v3/validation_report.json)。
候选ONNX：`experiments/transition_20261025_v3/candidate/policy.onnx`；原 `artifacts/running_policy.onnx` 不变。
Tracking动态AMP专家仍为原数据；新增的直跑保留教师是原交付跑步actor，仅用于训练，部署时不带教师。

本项目对应速度、起停转弯、域随机化、地形和持续奔跑目标。主机为 `robotit@100.101.111.80`，远端目录 `~/honor_ws/running_sim_pipeline`。新 Ubuntu 现已连接实机并完成部分只读诊断；本项目跑步策略仍只运行仿真与离线部署核对。官方 C++ 控制器的独立仿真键盘接入见 [official_policy_sim](../official_policy_sim/README.md)，本轮未把跑步 ONNX 直接替换到该控制器。

## 当前交付与结果（2026-10-20）

主策略是 [running_policy.onnx](artifacts/running_policy.onnx)，对应 [权重平均检查点](artifacts/averaged_4400_4600.pt)。它由 `rough_stability_seed42/model_4400.pt` 的 40% 与 `rough_precision_seed42/model_4600.pt` 的 60% 固定权重平均得到，不是多策略在线切换。来源哈希见同名 `.provenance.json`。该平均检查点只用于推理，禁止直接沿用其中的 PPO 优化器继续训练；续训应使用原始源检查点。

| 测试 | 主策略实际结果 | 结论 |
| --- | --- | --- |
| CPU 官方部署路径，名义平地，600 秒，指令 2.4 m/s | 无摔倒；稳态 2.315 m/s；含起跑全程前向均速 2.302 m/s；路径 1379.68 m | 通过速度与时长检查 |
| 崎岖地形 + 全部随机参数 + 观测噪声，3 个新种子 × 128 环境 × 25 秒，指令 2.2 m/s | 123/128、126/128、125/128；合计 374/384（97.40%）；各组稳态均速 2.049、2.065、2.069 m/s | 每组通过 ≥95% 与 ≥2 m/s 检查；仍有 10 次失败 |
| 名义平地 + 观测噪声，40 秒起停左右转，16 环境 | 16/16 完成；稳态 2.126 m/s；最差停止后速度 0.106 m/s、持续转向偏差 0.073 rad/s | 通过当前起停转弯检查 |
| 随机参数平地 + 观测噪声，40 秒起停左右转，32 环境 | 32/32 完成；稳态 2.087 m/s；最差停止后速度 0.165 m/s、持续转向偏差 0.124 rad/s | **未通过严格的停止/转向误差检查** |
| ONNX 与 PyTorch，运行中观测核对 | 最大动作绝对误差 1.91×10⁻⁶ | 通过导出一致性检查 |

完整检查见 [simulation_acceptance.json](artifacts/simulation_acceptance.json)，其中 `all_listed_simulation_checks_passed` 为 **false**。不能把不同模型的最好成绩合并成“全部达标”，也不能忽略失败试次。地形试次共享每个种子的两块合成地形，统计量不构成户外可靠性认证。600 秒连续跑是名义平地试验，地形测试另外进行，尚无同一次“崎岖地形 + 600 秒”的证据。

观看 [起停转弯视频](artifacts/running_transitions.mp4)、[崎岖地形代表性视频](artifacts/running_rough.mp4)、[地形失败示例](artifacts/running_rough_failed_env100.mp4)、[600 秒速度与路径图](artifacts/running_endurance.png) 和 [训练记录图](artifacts/training_evidence.png)。失败示例来自录像重复试验中的环境 100（约 6.2 秒失稳）；其余视频也另存重复试验记录，不替换上述预先选定的三组统计。代表性成功片段不能替代完整成功率。

CPU 600 秒的绝对机械功积分为 360.78 kJ，单位路径约 261.49 J/m。早期平地模型 `flat_799` 同为指令 2.4 m/s 时测得稳态 2.286 m/s、363.53 J/m，可作为历史对照；实际速度略不同，且未做算法消融，不能据此归因于某个奖励项或 AMP。这里的力矩峰值、RMS 和功率均按 50 Hz 采样；不是电池电耗或电机热测试。

实机关节方向、编码器零偏、连续力矩和实测 PD 仍为空值；[部署核对模板](assets/deployment_contract_template.json) 与 [未连接审计结果](assets/hardware_audit_unconnected.json) 已生成。600 秒试验没有航向保持闭环，零转向指令下仍会累计偏航，因此实际路径呈弧线；不能称为 600 秒精准直线跑。下一步仍需改善随机参数下的停止漂移、欠转与长时间航向漂移，再进行实机辨识与部署。本版本尚不能称为 sim2real 就绪。

## 控制方式

新任务直接接收身体坐标系下的 `[vx, vy, yaw_rate]`。它不会循环拼接 3.92 秒动作，也不会在动作末尾重置机器人姿态。机器人从名义站姿、零速度开始，之后由策略、PD、重力和接触推进。原有限片段 Tracking 用于构建 AMP 专家，不作为新任务的状态控制器。

- 实际工作空间的元气仔 MJCF，29 个关节，名义 PD 与力矩上限沿用官方资产。
- 该资产名义总质量为 41.56 kg，部分执行器峰值上限为 150 N·m；这些是仿真假设，不能与说明书约 35 kg、整机峰值描述直接等同。实机质量、逐关节连续/峰值能力和扭矩–速度曲线确认后，必要时应修正模型并再训练。
- 策略 50 Hz、物理仿真与 PD 200 Hz；动作是默认关节姿态加 `0.25 * actor_output`。
- 960 维 actor 输入，六类观测分别保留 10 帧历史：角速度、重力投影、速度指令、相对关节位置、关节速度、上一动作。MLP 为 512/256/128，初始权重来自官方 `loco.onnx`。
- 指令加速度上限：前向 0.8 m/s²、侧向 0.4 m/s²、转向 0.6 rad/s²；转弯时另限制指令向心加速度。
- 删除与跑步腾空冲突的 `fly` 惩罚；保留接触、姿态、滑脚、限位、动作变化与机械功率代价。

## 训练与 AMP

训练先按平地、随机参数平地、随机参数崎岖地形推进，随后延长高速指令保持时间，再加入摆脚净空与左右脚间距约束。脚本均在本目录，未修改官方训练仓库中的跟踪文件。净空奖励使用脚踝相对地形块原点的高度近似，不包含地形感知；左右脚横向间距在机体坐标中计算，不能用前后跨步长度替代。

| 随机项 | 当前训练范围 |
| --- | --- |
| 连杆质量 | 名义值 × 0.9–1.1 |
| 躯干质心 | 三轴各 ±15 mm |
| 足底摩擦 | 0.65–0.95 |
| PD 增益 | 名义值 × 0.85–1.15 |
| 执行器力矩能力 | 名义值 × 0.85–1.0 |
| 目标指令延迟 | 0–4 个物理步，即 0–20 ms |
| 推扰 | 每 7–12 秒，平面速度增量各 ±0.25 m/s |
| 观测噪声 | 沿用任务中的 IMU、重力投影、关节位置及速度噪声 |

推扰实现为瞬时速度增量，不是已标定的真实外力脉冲；延迟作用于执行器目标，不能替代真实通信与传感链路延迟测量。这些数值是当前仿真实验范围，尚未经硬件辨识。

AMP 使用相邻状态转移、判别器、策略回放缓冲、最小二乘分类损失及风格奖励，和 PPO 一起训练。特征包含关节位置/速度、重力投影、根高度、身体坐标速度和相对脚位置。早期平地阶段用原 GMR 参考及时间尺度增强；后续发现运动学与动力学数据差异导致判别器饱和，因此崎岖阶段改用经过验证的 Tracking 策略真实动力学样本，并加入标签平滑及插值点梯度惩罚。

`assets/dynamic_amp_expert.npz` 包含 32 段 Tracking 实际动力学轨迹采得的 5440 对转移，均无摔倒和参考回绕；来源、种子及 SHA256 见同名 JSON。专家采集允许参考状态初始化，学生跑步评估始终从静止站姿开始。不能把专家初始化和学生测试混为一谈。

历史交付实验未包含 AMP/PPO 消融；现已另行完成上文四组同预算三训练种子对照，仍不能把历史性能变化全部归因于 AMP。

## 检查协议

`evaluate.py` 记录真实仿真速度、完成率、停止后速度、转向误差、倾斜、机械功率积分、逐关节力矩 RMS 和腾空比例。自动重置后仍将该试次记为失败。误差和能耗必须和完成率一起阅读。

- 起停转弯：40 秒，包含站立、加速跑、正向转弯、停止、反向转弯、再次停止。
- 连续跑：600 秒，从静止起跑；另使用官方 `LocomotionState` 与 `MuJoCoRobot` 的独立 CPU 部署路径核验。
- 崎岖地形：合成高度噪声 −15 至 +25 mm、波浪幅度配置 15–40 mm，两类 160×160 米地形；根部离开自身测试地形时记为越界失败。合成地形不代表所有户外崎岖路面。
- 平地无观测噪声时是重复名义起始状态，不应称为独立扰动样本。随机参数/地形、种子和 `--noise` 条件分别保存在结果中。
- 工程速度判据为完成率至少 95%、实际稳态均速至少 2 m/s；600 秒测试另报告含起跑的全程前向均速和路径长度/时间。
- 起停转弯检查停止后速度不超过 0.15 m/s、0.2 秒一阶低通后的转弯角速度 MAE 不超过 0.15 rad/s、左右转各自平均有符号偏差不超过 0.08 rad/s、基座倾斜 RMS 不超过 0.25 rad。原始未滤波 MAE 和原始判据 `passes_raw_transition_checks` 同时保留，不能把滤波通过描述为原始振动消失。该协议用于区分步态周期振动与持续转向不足，工程阈值不替代课程实机验收。

`sum(abs(tau * dq))` 是机械功率代理，按采样间隔积分，不是电池侧功耗。仿真没有已标定电机热模型、驱动效率或电池模型，600 秒仿真不等于实机 10 分钟续航或低温升验收。

## 运行

所有命令在远端现有环境执行：

```bash
cd ~/honor_ws/honor_rl_mjlab
PY=.venv/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "$PY" ../running_sim_pipeline/train.py \
  --stage flat --expert-source kinematic --num-envs 1024 --iterations 800 \
  --save-interval 200 --log-dir ../running_sim_pipeline/logs/reproduce_flat
```

随机参数阶段接续同架构的检查点，保留 PPO/优化器及 AMP 状态：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "$PY" ../running_sim_pipeline/train.py \
  --stage robust --expert-source kinematic --num-envs 1024 --iterations 800 \
  --resume ../running_sim_pipeline/logs/reproduce_flat/model_799.pt \
  --log-dir ../running_sim_pipeline/logs/reproduce_robust
```

更换为动力学专家时明确重置判别器，避免把旧数据的归一化参数误用到新数据：

```bash
"$PY" ../running_sim_pipeline/collect_dynamic_expert.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "$PY" ../running_sim_pipeline/train.py \
  --stage rough --expert-source dynamic --reset-amp --num-envs 1024 --iterations 1400 \
  --resume ../running_sim_pipeline/logs/reproduce_robust/model_1598.pt \
  --log-dir ../running_sim_pipeline/logs/reproduce_rough
```

随后两轮各追加 800 次更新，均保持动态 AMP 专家与优化器状态：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "$PY" ../running_sim_pipeline/train.py \
  --stage rough --expert-source dynamic --focus-running --num-envs 1024 --iterations 800 \
  --resume ../running_sim_pipeline/logs/reproduce_rough/model_2997.pt \
  --save-interval 200 --log-dir ../running_sim_pipeline/logs/reproduce_focus
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "$PY" ../running_sim_pipeline/train.py \
  --stage rough --expert-source dynamic --focus-running --refine-stability \
  --num-envs 1024 --iterations 800 \
  --resume ../running_sim_pipeline/logs/reproduce_focus/model_3796.pt \
  --save-interval 200 --log-dir ../running_sim_pipeline/logs/reproduce_stability
```

`--focus-running` 将移动样本中的 75% 设为 2–2.6 m/s、65% 设为直线，指令保持 7–14 秒，仍保留 20% 站立样本，回合时长 30 秒。`--refine-stability` 增加摆脚净空不足代价（脚踝高度目标 0.14 m，按水平脚速加权）、横向脚间距不足代价（0.15 m）、收紧 yaw 奖励 std 至 0.3 并将其权重设为 6，摔倒惩罚改为 −100。这些训练超参数不是硬件规格。

训练的 `--iterations` 是追加数量。五段实际共追加 4600 次更新；框架续训从存档迭代标签开始计数，因此最终文件标签为 4595。配置、专家来源和后续训练的源码快照保存在各日志目录。评估种子不能简单等同于独立训练种子：本轮只有训练种子 42，没有多训练种子消融。

随后从稳定性阶段的 4400 检查点追加精调：`--precision-running` 将脚踝净空目标提高到 0.17 m、净空代价权重设为 −30，加入 0.2 秒低通后的持续转向跟踪奖励（实际与目标同滤波，std 0.2、权重 4），瞬时转向奖励权重改为 2.5，摔倒代价恢复 −200。该轮计划追加 800 次，在日志标签 4819 处停止，实际追加 420 次；保留了 4600、4800 检查点及 `stop_record.json`，未把未完成的计划次数记为已训练次数。六阶段正式训练累计更新 5020 次（不含小规模运行检查），交付平均策略使用其中的 4400 与 4600 两个源检查点。

重建精调源与平均候选的命令如下；已有 `assets/dynamic_amp_expert.npz` 时使用附带数据，重新采集会改变 SHA，不能与原 AMP 状态混用。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 "$PY" ../running_sim_pipeline/train.py \
  --stage rough --expert-source dynamic --focus-running --refine-stability --precision-running \
  --num-envs 1024 --iterations 201 --save-interval 200 \
  --resume ../running_sim_pipeline/logs/rough_stability_seed42/model_4400.pt \
  --log-dir ../running_sim_pipeline/logs/reproduce_precision
"$PY" ../running_sim_pipeline/average_actor.py \
  --first ../running_sim_pipeline/logs/rough_stability_seed42/model_4400.pt \
  --second ../running_sim_pipeline/logs/reproduce_precision/model_4600.pt \
  --second-weight .6 --output ../running_sim_pipeline/artifacts/reproduced_average.pt
```

GPU 接触仿真存在非确定性，同一种子重复运行也可能有不同失稳试次；精确文件、版本与源码哈希见 `artifacts/environment_and_source.json` 和 `delivery_manifest.json`。完整训练日志与未交付的中间检查点保留在远端 `logs/`，本地 `configs/` 保存各阶段配置及可获得的源码快照，清单见 `configs/index.json`。

评估与独立 CPU 部署：

```bash
"$PY" ../running_sim_pipeline/evaluate.py --checkpoint /absolute/path/model.pt \
  --stage rough --scenario straight --speed 2.4 --duration 25 --num-envs 32 --seed 456 --noise \
  --output ../running_sim_pipeline/results/custom_rough.json
"$PY" ../running_sim_pipeline/evaluate_cpu.py --policy /absolute/path/policy.onnx \
  --speed 2.4 --duration 600 --output ../running_sim_pipeline/results/custom_cpu_600s.json
```

## 实机标定准备

`assets/deployment_contract_template.json` 给出 29 关节名称、仿真名义姿态、PD、峰值力矩和限位；SDK 索引、编码器方向、零偏、真实连续/峰值能力及实测 PD 均保持待测，不能把仿真默认站姿当作编码器零位。对已经标定的 SDK 物理角度不能重复去零偏。

`deployment_audit.py` 可以离线核对实测表与已训练范围、连续力矩 RMS 和同一主机时钟下的发送/应答延迟日志。延迟 CSV 列为 `send_monotonic_ns,ack_monotonic_ns`，至少 1000 条有序样本。审计脚本没有向机器人发命令的功能，输出也不构成实机部署批准。

```bash
"$PY" ../running_sim_pipeline/deployment_audit.py \
  --policy ../running_sim_pipeline/artifacts/running_policy.onnx \
  --measurements measured_calibration.json --evaluation results_600s.json \
  --latency communication_latency.csv --output calibration_audit.json
```

后续实机阶段仍需厂家支持的部署接口、扭矩-速度曲线、热降额、IMU 外参及电池侧电压/电流实测。
