# 元气仔导航仿真维护说明

本目录是仿真适配模块。默认只允许127.0.0.1、ROS_LOCALHOST_ONLY=1和独立ROS域。
不要导入宇树SDK、下发LowCmd或加入任何实机控制入口。

位移必须来自ONNX策略、官方观测/关节PD部署路径和MuJoCo接触。禁止在运行循环中写浮动基座位姿或速度。
地图必须由slam_toolbox在线生成；不得使用空白地图或静态map→odom变换冒充SLAM。
默认使用官方行走策略；用户指定的跑步演示使用running_sim_pipeline/artifacts/running_policy.onnx，必须记录策略哈希与实际速度。

测试入口为scripts/run_test.sh，结果必须同时检查integration.json和simulation.json，
不能只凭轨迹图、视频或到达标记宣布通过。报告必须说明理想里程计与合成激光的范围。
源代码修改后先运行tests/test_protocol.py和ROS系统环境的tests/check_ros_algorithms.py。
需要检验底层控制时运行scripts/smoke_control.py。不要提交build/install/log、缓存或视频。

从_g1复制的源码保持不改，通过navigation.py适配；保留原BSD许可与来源哈希。
