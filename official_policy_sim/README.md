# 官方部署控制器：新 Ubuntu 仿真键盘验证

本模块把 HonorRobotics 的 `honor_rl_deploy` C++ 控制器编译到新 Ubuntu（`s@100.88.187.23`），通过 ROS 2 连接本机 MuJoCo，验证官方行走 ONNX 的键盘控制链路。它与此前 `robot_keyboard_control` 调用机器人原厂高层 SDK 的方式不同。

**本模块只启动仿真。没有实体机器人启动选项，不修改机器人服务，不切换调试模式，也不向实体机器人发送指令。** 新 Ubuntu 上的工作目录为 `~/honor_deploy_ws`；原厂硬件链路的 ROS domain 为 0，本模块固定使用 domain 117、仅回环网卡的 Fast DDS、自定义 `/honor_sim_only/` 话题。

## 使用

在新 Ubuntu 的**桌面终端**执行以下命令，打开仿真窗口并用这个终端的键盘控制模型：

```bash
bash ~/honor_deploy_ws/official_policy_sim/run_smoke.sh --manual
```

1. 保持键盘焦点在启动程序的终端，按一次 `R`，等待至少 4 秒，再按一次 `L`。
2. 看到 `Free dynamics started` 后，模型开始自由动力学仿真。此时可按住或连续按 `W/S` 前进/后退、`A/D` 横移、`Q/E` 转向。
3. 松键后，400 ms 的按键保持窗口到期，输入回零，再经过官方指令滤波。不是瞬时机械制动。
4. `Ctrl+C` 退出本次仿真及其 C++ 控制器。官方的空格键是 Damper，`P` 是 Passive；这两个模式不维持主动平衡，不应当作“停稳站立”键。

仿真键盘上限：前后向 0.5 m/s、侧向 0.3 m/s、转向 0.35 rad/s。这些是测试配置，不是实机限速或能力认证。输入终端与 MuJoCo 窗口是两个不同焦点，按键需发送到终端。

通过 SSH 重跑自动测试（不需要显示器）：

```bash
bash ~/honor_deploy_ws/official_policy_sim/run_smoke.sh
```

自动测试向**仿真控制器的伪终端**依次输入 R、L、前进、松键、左转、松键。结果写入 `~/honor_deploy_ws/results/smoke_日期时间/`。它不是一条实机测试命令。

## 测试边界

初始状态为名义站姿、零速度，足底放置在地面上方 2 mm。反馈在 R/等待/L 的初始化阶段冻结，进入行走策略 0.7 秒后释放物理仿真。释放后不使用弹性吊带、外部支撑力、根部重置或姿态传送。因此这项测试检验**从名义站姿进入闭环后的站立、行走和按键响应**，不证明机器人可以从软关节或任意蹲姿自主恢复。

仿真使用官方 `honor_mujoco` 的 29 自由度资产；质量 41.56 kg。只保留资产原有的一个地面，摩擦配置为 `0.8 0.02 0.001`；逐关节峰值力矩按本项目 `running_sim_pipeline/assets/deployment_contract_template.json` 中的仿真假设限制。它们不是已完成的实机标定。

第一次原型测试错误地重复添加地面，已撤出最终证据。修正后的重测以 [测试报告](TEST_REPORT.md) 和对应 `result.json` 为准，不能沿用第一次的数字。自动测试仅包含短时间平地试次，不是速度、地形、续航或 sim2real 验收。

## 你的跑步模型是否在这个基础上训练

是建立在官方机器人资产和官方行走 actor 的基础上继续训练，但需要区分两个工程：

| 项目 | 本项目实际用途 |
| --- | --- |
| `honor_rl_mjlab` | PPO 强化学习训练；从 `deploy/checkpoints/loco.onnx` 加载 actor，之后加入 LS-AMP、随机化和跑步任务 |
| `honor_rl_deploy` | C++ 状态机、键盘输入、观测历史、ONNX 推理和关节命令；本轮验证的是该仓库自带的行走模型 |

训练记录 `logs/flat_v2_seed42/warm_start_validation.json` 确认了官方初始化，加载后的动作最大绝对误差约 `7.11e-4`，输入为 960 维，critic 新建。当前跑步交付策略仍是 `running_sim_pipeline/artifacts/running_policy.onnx`，本轮未替换它。

**训练初始化的 ONNX 与 C++ 仓库自带的 ONNX 权重不同。** 两者都是 `960 → 29`，但 50 个相同随机输入的输出最大差值约 11.62。更关键的是，C++ 默认策略使用交错关节重排，而本项目跑步 ONNX 的关节顺序是左腿、右腿、腰、左臂、右臂。在本轮仿真执行器顺序下，跑步策略应使用 0–28 的恒等映射。完整哈希、原始重排、元数据和核对方法见 [model_comparison.json](model_comparison.json)。

因此不能只把 `running_policy.onnx` 改名覆盖。下一步接入跑步策略时，需要另建配置并核对关节映射、六类观测各自的 10 帧历史、动作比例 0.25、默认姿态、PD、50 Hz 策略频率、物理步长和仿真资产差异，然后重新做 C++ 闭环验证。本轮未声称完成了这项跑步策略适配。

机器人本体预装的原厂策略是否与公开文件相同，当前没有证据。本轮仿真成功也不能解释或解决此前实机的关节反馈异常；实机独占控制、驱动/动力就绪、零位方向和保护流程仍需分别确认。

## 来源与构建

| 官方工程 | 本轮固定提交 |
| --- | --- |
| [honor_rl_deploy](https://github.com/HonorRobotics/honor_rl_deploy) | `709c809d98ada109f8d5587ea655f1d4ca89ce47` |
| [honor_mujoco](https://github.com/HonorRobotics/honor_mujoco) | `7c76b0cc6d7b706e5726d4acfbe4e5fa75267a89` |
| [honor_robot_sdk](https://github.com/HonorRobotics/honor_robot_sdk) | `43f1dba12e2e0d25aaaec0fe4978da179a8c26f2` |

从 4080 上已有的官方检出目录执行 `git archive HEAD`，经 Windows 传到新 Ubuntu，核对归档 SHA-256 后解压。未拷贝旧机器的虚拟环境或编译产物。归档路径为新 Ubuntu 的 `~/honor_deploy_ws/archives/`，源码在 `src/`。上游许可证保留在各源码目录。

新 Ubuntu 增装了 `python3-venv libgoogle-glog-dev libglfw3 xvfb`。ROS 包源下载超时后，没有修改系统软件源；改在独立 Python 3.10 venv 内安装 colcon。已存在的 Eigen、yaml-cpp、OpenCV、ROS 2 Humble 沿用本机安装。CMake 移除了源码中没有使用的 `nav2_util` 依赖；C++ 控制逻辑未修改。`V1` 原配置保留，另建 `SIM_ONLY`。

当前环境重编译：

```bash
bash ~/honor_deploy_ws/official_policy_sim/build_ubuntu.sh
```

文件用途：`prepare_sources.py` 校验归档并生成配置；`sim_env.sh` 与 `localhost_dds.xml` 约束通信；`run_controller_sim.sh` 在启动前再次检查仿真话题和标志；`smoke_keyboard.py` 复用官方桥接序列化和 PD 计算，并管理仿真、终端和退出；`render_trial.py` 仅回放保存的状态，不运行 ROS 或控制器。

这不是可直接复制到机器人主板上运行的实机部署脚本。

官方行走模型已在机器人独立目录完成 ARM64 编译与离线推理，见 [机载部署和操作说明](onboard/README.md)。未启动实机控制器；8 个关节的通信应答仍异常，启动检查会拒绝运行，尚未验证实机运动。
