# 元气仔：从 G1-Nav 迁移的 SLAM 与导航仿真

已新增**之前训练的跑步策略 + 在线SLAM + RViz同步录像**，三目标测试通过。先看 [双画面演示视频](results/running_demo/running_slam_rviz_demo.mp4) 与 [最新测试记录](TEST_REPORT.md)。运行 `bash scripts/run_running_demo.sh`；完整输出在独立的 `results/running_demo_*` 目录。本次交付保存在 `results/running_demo/`，下面原有内容主要描述默认的官方行走模式。

将原 `G1-Nav-master` 的全局路径规划和局部导航前端接到元气仔官方速度控制策略，使用真实关节与 MuJoCo 接触动力学产生位移，并运行 ROS 2 Humble 的 `slam_toolbox` 在线建图。

当前只支持仿真。此模块不导入宇树 SDK，不发布 LowCmd，也不连接实机网卡。UDP 只绑定 `127.0.0.1:15100/15101`，ROS 默认 `ROS_LOCALHOST_ONLY=1`、独立域 `ROS_DOMAIN_ID=91`。

## 技术路线

```text
RViz目标 /g1pilot/goal
    → G1-Nav Dijkstra规划：SLAM地图、未知区域拒绝、0.65m膨胀、拐角碰撞检查
    → 元气仔适配的路径跟踪：先转向、再前进、低速指令补偿、局部制动
    → /g1pilot/auto_joy → 本机UDP
    → 元气仔官方 loco.onnx → 29关节目标与PD → MuJoCo接触动力学
    → 实际关节/位姿 + 同一物理场景的模拟激光
    → ROS里程计/TF/scan/clock → slam_toolbox → /map 与 map→odom
```

原 `navigation_mjlab.launch.py` 的默认前端是预设空白地图，本模块重新接入在线 `slam_toolbox`。运行时没有空白地图发布者，也没有伪造的静态 `map→odom`；规划和跟踪都会使用 SLAM 的实际 `map→odom` 变换。

## 已实现的迁移

| G1相关部分 | 元气仔适配 |
| --- | --- |
| 宇树G1模型、29关节策略、SDK/DDS控制器 | Honor官方模型、关节顺序、PD与ONNX观测历史 |
| 固定G1关节列表与基座高度 | 关节名称由官方部署接口传入，基座高度由物理状态测量 |
| 导航与SLAM坐标系混用 | 将实际里程计转换到地图坐标后再规划与跟踪 |
| 几何公式模拟扫描 | 对MuJoCo中实际存在且可碰撞的墙体做360束射线检测 |
| 低速比例跟踪 | 增加元气仔策略的低速补偿，明显偏航时先原地转向 |
| 按扫描束数累加的排斥速度 | 改为对障碍物的接近速度制动，避免扫描密度导致过强侧推 |
| 路径捷径检查 | 补充supercover检查，禁止从障碍格拐角切过 |
| 通信与数据失效 | 命令、位姿、扫描、路径和TF新鲜度检查；250ms通信看门狗 |

原全局规划与跟踪代码快照保存在 `vitaboy_navigation/_g1/`，上层适配在 `navigation.py`。原G1工程没有被改动。

## 环境与目录

已在 `robotit@100.101.111.80`（主机名4080，RTX 4080 SUPER）上开展仿真联调。

```text
~/honor_ws/
├── honor_rl_mjlab/          # 官方框架，固定10c2ba02...
│   └── .venv/              # Python 3.12，MuJoCo 3.10，ONNX运行时
├── honor_robot_description/# 官方VitaBoy URDF和网格
└── vitaboy_navigation/      # 本模块；ROS节点使用系统Python与Humble
```

物理控制与策略使用官方环境，ROS使用系统环境，两者通过UDP隔离，避免Python 3.12与ROS Humble系统Python的依赖冲突。默认策略是官方行走 `loco.onnx`，不是高速跑步模型。

## 构建与自动测试

在4080的Ubuntu终端执行：

```bash
cd ~/honor_ws/vitaboy_navigation
bash scripts/build.sh
bash scripts/run_test.sh
```

自动测试启动独立仿真和ROS进程组，等待在线SLAM地图后，依次发布 `(0.8,0)`、`(0.5,1.2)`、`(2.8,0)` 三个目标，再发送障碍物内/未知区域目标验证拒绝与停车。结束时只关闭本脚本创建的进程。

默认速度上限为前向 `0.35 m/s`、侧向 `0.15 m/s`、转向 `0.35 rad/s`；前向加速度限制 `0.8 m/s²`。这些是本次仿真配置，不是实机规格。规划膨胀半径 `0.65m` 为当前保守设置，还需结合实际外形与摆动范围校准。

测试产物：

- `results/integration.json`：SLAM、逐目标结果、话题频率、拒绝目标与停车检查。
- `results/simulation.json`：实际物理轨迹、摔倒/障碍物接触、策略哈希与无支撑标记。
- `results/integration.map.json`：实际SLAM栅格与地图坐标下的机器人轨迹。
- `results/navigation.mp4`：同一测试中实际MuJoCo运行画面。
- `results/slam_navigation.png`：SLAM地图与实测导航轨迹。
- `results/control_smoke.json`、`control_smoke.mp4`：独立22秒站立、行走、停止与转向测试。

结果图生成：

```bash
../honor_rl_mjlab/.venv/bin/python scripts/plot_results.py --results results
```

早期失败结果以 `*_failure.json` 等名称保留，记录低速指令、避障增益和规划间距适配中发现的问题；它们不能与最终结果混用。

## 带窗口交互测试

在4080的图形桌面终端中执行：

```bash
cd ~/honor_ws/vitaboy_navigation
bash scripts/run_interactive.sh
```

有可用显示会话时会打开MuJoCo与RViz；在RViz使用 **2D Goal Pose** 设置目标。RViz加载元气仔实际URDF，显示SLAM地图、路径、激光、TF和机器人。没有图形会话时运行无窗口模式。

如目录不同，可设置 `HONOR_REPO=/absolute/path/honor_rl_mjlab` 和 `HONOR_URDF=/absolute/path/vita_boy_29dof_v1.urdf`。本机端口被占用会直接报错，不会自动杀掉其他仿真。

命令行设置目标：

```bash
source /opt/ros/humble/setup.bash
source ~/honor_ws/vitaboy_navigation/install/setup.bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=91
ros2 topic pub --once /g1pilot/goal geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 0.8, y: 0.0}, orientation: {w: 1.0}}}'
```

SLAM当前只认识扫描过的区域；未知目标会被拒绝。跑步策略已完成上方所述单场景联调；自动探索、动态行人、复杂地形和实机导航尚未验证。

## 验证边界

本次使用由物理引擎实际状态产生的**理想里程计**，以及水平稳定、无遮挡自车的理想二维扫描。激光射线与碰撞障碍物来自同一场景。测试说明SLAM、规划、跟踪和元气仔实际步行控制可以连通，不等于验证了真实雷达、腿式里程计、IMU融合或定位漂移。

运行中不写浮动基座位姿或速度，不使用虚拟吊带，不在目标处重置机器人。只有加载仿真时设置初始站姿。渲染视频来自同次物理运行。

真实部署前仍需要传感器与IMU外参、可靠里程计、机器人外形与制动距离、通信延迟和官方策略接口等适配。本模块没有实机下发入口。

## 来源与许可

- 原导航工程：本地 `G1-Nav-master`，BSD-3-Clause。原始文件哈希见 `provenance.json`，许可证见 `LICENSE`。
- [HonorRobotics/honor_rl_mjlab](https://github.com/HonorRobotics/honor_rl_mjlab)，固定提交 `10c2ba02c7c001bb574771855b9dc41ef193e01d`。仿真执行器构建沿用该提交的部署实现，原许可见 `THIRD_PARTY_HONOR_LICENSE`。
- [slam_toolbox](https://github.com/SteveMacenski/slam_toolbox)：在线二维SLAM，使用Humble已安装软件包。
- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html)：物理仿真和射线接口。
