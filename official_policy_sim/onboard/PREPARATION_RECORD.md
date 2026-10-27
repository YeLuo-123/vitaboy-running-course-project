# 历史记录：首次连接中断时的机载部署准备

本页保留首次准备时的状态；重启后的实际部署结果及操作命令见 [README.md](README.md)。

日期：2026-10-26。用户指定使用官方行走模型，并要求保留机器人原有程序。本轮完成兼容性检查和独立部署包准备，**尚未成功复制到机器人，也未完成 ARM64 编译或实机控制验证**。

## 结论与当前进度

官方 `honor_rl_deploy` 支持机载电脑部署和键盘输入。ONNX 可以放在独立目录，由独立控制器加载；不必覆盖机器人预装的程序或模型。不过实际运行时需要按厂家流程交接关节控制权，不能让两个控制器同时发布关节指令。启动控制器本身就可能发布 LowCommand，即使还没按 W，也不能称为“无运动测试”。

本轮没有执行调试模式切换，没有停止、禁用或重启原厂服务，没有删除或覆盖机器人文件，没有启动实机控制器，也没有发送关节、姿态、速度或电源指令。

| 检查 | 实际结果 |
| --- | --- |
| 主板登录 | 曾成功登录 `hihonor@192.168.42.201` 并只读获取环境，后续连接中断 |
| 架构 | 主板 aarch64、Ubuntu 22.04；外部主机 x86_64，现有外部主机二进制不能直接复制使用 |
| 编译依赖 | 主板已有 cmake、g++、make、Eigen、yaml-cpp、glog、OpenCV、curl、zlib；无需为本轮准备升级系统 |
| SDK | 沿用主板 `/opt/build_dist/common`；LowCommand、LowState、MotorCommand、MotorState 的去注释定义与官方源码一致 |
| 模型 | 官方部署仓库 `vita_boy_loco.onnx`，SHA-256 为 `c8978c07a7ce2cd85370ff1010cc23a1b2ec0a0686b024fc7533e947852d19a7` |
| 离线检查程序 | 已在外部 x86 Ubuntu 编译、运行；输入 1×960，输出 1×29，520 次推理均有限值，其中 500 次计时。该结果不代表 ARM 主板性能 |
| 机载复制/编译 | 未完成；创建机载目录的 SSH 尝试未成功，未传入源码或模型 |
| 运行/行走 | 未执行 |

检查时主板 `motion_intelligence.service` 的 `LoadState=not-found`，不是一个已加载后停止的服务。不能只看到 `ActiveState=inactive` 就断言原厂控制器已退出；原厂控制可能由其他服务、进程或另一控制板管理。后续必须结合实际 ROS 关节指令发布者和厂家调试流程确认。

证据：[准备状态](evidence/onboard_preparation_status.json)、[主板环境快照](evidence/onboard_before.json)、[SDK 对比](evidence/sdk_interface_comparison.json)、[外部主机离线推理](evidence/offline_probe_x64.json)。这些不是实机运动通过记录。

## 当前连接阻塞

新 Ubuntu 的 r8169 网卡内核日志多次记录 `Link is Down`、重新连接，以及 `Downshift occurred from negotiated speed 1Gbps to actual speed 100Mbps, check cabling!`。机器人保持开机、网线已连接，但 SSH 后续出现连接拒绝、超时和无路由。日志不能区分最终原因是线缆、接头、网口还是驱动；不能仅凭降速确定某根线损坏。

仅在外部主机试验了 100 Mbps 协商限制，未恢复可靠连接，已撤回该速率限制。没有改动机器人网卡配置。恢复可靠有线连接后，才继续传输和编译；当前链路不能作为键盘控制的可靠基础。

## 外部 Ubuntu 的可恢复网络调整

原 `wheeltec-wired-dhcp` 配置保留。新建 `honor-robot-192.16842`，用于 `enp3s0`：

- 手动地址 `10.42.0.1/24` 与 `192.168.42.100/24`，不增加默认网关。
- 自动连接优先级 100，避免网线断开重连后丢失机器人网段地址。
- 新配置采用 manual，不提供旧 shared 配置的 DHCP/NAT 服务；旧配置仍可切回。Wi-Fi 和 Tailscale 配置未修改。
- 试验后的链路设置恢复为 auto-negotiate=no、speed=0、duplex 未指定，即不强制更改原链路协商模式。

如要切回此前的有线配置，可在外部 Ubuntu 执行：

```bash
sudo nmcli connection modify honor-robot-192.16842 connection.autoconnect no
sudo nmcli connection up wheeltec-wired-dhcp
```

这只影响外部主机的有线连接，不删除任何配置。

## 独立目录与文件

拟定机载目录（本轮尚未建立成功）：

```text
/data/hihonor_onnx_keyboard/official_walk_709c809_20261026/
  honor_rl_deploy.tar.gz          固定提交的官方源码归档
  onboard/                       本目录的构建、离线检查脚本
  honor_rl_deploy/                独立解压、独立编译
  tools/onnx_cpu_probe            不链接 ROS 的离线推理程序
  logs/                          构建与离线检查结果
```

外部主机现有材料在 `~/honor_deploy_ws/archives/`、`~/honor_deploy_ws/onboard_tools/`。其中官方源码归档为 `honor_rl_deploy.tar.gz`，工具归档为 `onboard_kit.tar.gz`；本地源码更新后应重打工具归档再传输。原厂程序目录、systemd 单元、开机配置和 SDK 安装目录不在写入范围内。

`prepare_onboard.py` 只允许在上述 `/data/hihonor_onnx_keyboard/` 的新版本目录执行，核验归档与模型 SHA，保留 V1，创建 ONBOARD_WALK 配置。该配置限前后向 0.3 m/s、横向 0.25 m/s、转向 0.3 rad/s，并禁用未测试的 Tracking 状态。配置数值不是实机标定结论。

`build_onboard.sh` 在该新目录里构建 ARM64 控制器和离线检查程序，使用机器人已有的 SDK 和库，不安装系统包，不运行完整控制器。`start_build_detached.py` 仅启动上述构建脚本。**本目录尚不提供实机启动脚本。** 只有机载编译、离线验证、真实状态/关节顺序与控制权检查完成后，才准备现场操作步骤。

源码依据：[HonorRobotics 部署说明](https://github.com/HonorRobotics/honor_rl_deploy/blob/709c809d98ada109f8d5587ea655f1d4ca89ce47/readme_zh.md)。网络速率测试依据：[NetworkManager 网卡设置说明](https://networkmanager.dev/docs/api/latest/settings-802-3-ethernet.html)。
