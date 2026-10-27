"""Write a technical audit from measured experiment JSON files."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--final',type=Path,required=True)
    p.add_argument('--middle',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    report=read(a.final/'validation_report.json')
    first=read(a.first/'validation_report.json')
    baseline=read(a.final/'baseline/simulation_acceptance.json')
    runs=sorted(set([a.first]+([a.middle] if a.middle else [])+[a.final]),key=lambda r:r.name)
    audits=[read(r/'training_audit.json') for r in runs]
    legacy=next(t for t in report['randomized_transitions'] if 'legacy_transitions' in t['file'])
    new=[t for t in report['randomized_transitions'] if 'final_transitions' in t['file']]
    before=report['baseline_paired_transitions']
    worst=lambda rows,key:max(r[key] for r in rows)
    cpu=report['cpu_endurance']
    accepted=report['all_checks_passed']
    verdict='候选通过本轮全部预先声明的检查，可作为下一阶段仿真候选。' if accepted else '起停转向改善，原有三组地形回归和600秒速度得到保留；新增地形泛化保留检查仍未通过，因此总体未达全部要求，默认交付策略保留原版。'
    source=read(a.final/'selection.json')
    interpolation=Path(source['source_checkpoint']).name.startswith('blend_')
    text=f'''# 随机条件下起停与转向稳定性专项优化记录

日期：2026-10-25。仅仿真，无实机执行。

## 结论

{verdict}

原 `artifacts/running_policy.onnx` 保留，SHA-256：`{baseline['policy']['sha256']}`。
本轮候选单独导出，策略SHA-256：`{report['candidate_policy']['sha256']}`。

## 训练与筛选

- 第一段追加800次PPO更新：一半环境练习高速直跑等原有指令，一半练习站立、左右转与两次减速停止；强化持续航向跟踪与停稳奖励。
- 第一段候选完整验证未通过：600秒全程均速{first['cpu_endurance']['whole_trial_forward_speed_mps']:.3f}m/s，原地形{first['legacy_terrain_successes']}/384；两个新种子的停车速度略超0.15m/s。所有失败记录保留。
- 第二段追加400次：速度奖励权重6、std 0.45；停稳奖励权重5，静止漂移代价权重−6。继续使用随机参数崎岖地形和原动态AMP专家。
- {'第二段最佳检查点的开发地形完成123/128，同期基线122/128、历史基线124/128；各轮同一基线重复结果为122—125/128，不能凭单次少量样本差异确认泛化保持。两个权重插值候选未通过开发检查。第三段追加200次，在高速直跑状态加入冻结基线actor的训练期蒸馏约束，站立和转向状态不加此约束；部署时不带教师。' if a.middle else '候选均按开发集的起停、速度和地形联合表现筛选。'}
- 共计{sum(r['formal_updates'] for r in audits)}次正式更新、{sum(r['control_transitions'] for r in audits):,}条控制转移；训练进程耗时合计{sum(r['training_process']['elapsed_s'] for r in audits)/60:.2f}分钟，不含{sum(r['smoke_updates_excluded'] for r in audits)}次冒烟更新和评估耗时。
- 1024并行环境，策略50Hz、物理/PD 200Hz；固定学习率分别为{', '.join(str(r['optimizer_lrs'][0]) for r in audits)}，从原始有效检查点续训，未使用平均模型的优化器。
- 第一段验证暴露的种子105261、105262随后明确转为开发集。最终验证的新种子为105271、105272、105273，未参与候选选择。
- 选择来源：`{source['source_checkpoint']}`。
- {'该候选由两个已训练actor固定权重插值得到，运行时只有一个策略；检查点仅供推理，不得直接恢复其中优化器。续训使用两个原始源检查点。' if interpolation else '候选来自原始训练检查点；执行时不在线混合策略。'}

## 同一候选的实测回归

| 项目 | 原基线 | 本轮候选 |
| --- | --- | --- |
| 原随机种子718281：最差停车速度 | 0.1651m/s | {legacy['stop']:.4f}m/s |
| 原随机种子718281：最差左右转持续偏差 | 0.1243rad/s | {legacy['bias']:.4f}rad/s |
| 新三种子：最差停车速度 | {worst(before,'stop'):.4f}m/s | {worst(new,'stop'):.4f}m/s |
| 新三种子：最差左右转持续偏差 | {worst(before,'bias'):.4f}rad/s | {worst(new,'bias'):.4f}rad/s |
| 原三地形种子完成数 | 374/384 | {report['legacy_terrain_successes']}/384 |
| 新三地形种子完成数（配对） | {report['paired_baseline_terrain_successes']}/384 | {report['new_terrain_successes']}/384 |
| CPU名义平地600秒全程前向均速 | {baseline['cpu_endurance']['whole_trial_forward_speed_mps']:.3f}m/s | {cpu['whole_trial_forward_speed_mps']:.3f}m/s |
| CPU机械功代理 | {baseline['cpu_endurance']['absolute_mechanical_cost_jpm']:.2f}J/m | {cpu['mechanical_cost_per_meter_jpm']:.2f}J/m |

新三种子各32环境，40秒起停左右转；地形各128环境、25秒。失稳后自动重置仍计失败。原地形种子718281/828182/938293只作回归，不能当作本轮未见地形。

## 固定检查与保留条件

沿用原 `evaluate.py` 和 `evaluate_cpu.py`，未放宽阈值，未改变测试控制器、初始化、地形或随机参数：

- 起停转向：完成率≥95%、稳态均速≥2m/s，最差停车速度≤0.15m/s、左右转持续偏差≤0.08rad/s、0.2秒低通角速度MAE≤0.15rad/s、机身倾斜RMS≤0.25rad。
- 保留原始未滤波角速度误差和动作变化率；低通指标通过不等于高频振动消失。
- 原地形总完成数≥374/384，各种子完成率≥95%、均速≥2m/s，并与原速度对照；新地形总完成数不得低于配对基线。
- 训练前声明速度允许最多0.05m/s的重复仿真差异，且始终≥2m/s；CPU600秒全程均速保留门槛2.2519m/s。不是宣称与基线速度完全相等。
- 同一个检查点完成全部GPU检查，同一导出ONNX完成CPU检查，均核对SHA-256；无虚拟支撑、无参考基座传送。

| 检查 | 结果 |
| --- | --- |
'''
    for name,passed in report['checks'].items():
        text+=f"| `{name}` | {'通过' if passed else '未通过'} |\n"
    text+='\n## 新地形逐组结果\n\n| 种子 | 基线完成数 | 候选完成数 | 候选均速 | 最少稳态采样数 | 完整判据 |\n| --- | --- | --- | --- | --- | --- |\n'
    for row,b in zip(report['new_terrain'],report['baseline_paired_terrain']):
        raw=read(a.final/'results'/Path(row['file']).name)
        text+=f"| {raw['seed']} | {round(b['completion']*128)}/128 | {round(row['completion']*128)}/128 | {row['speed']:.3f}m/s | {int(min(raw['steady_samples_each_env']))} | {'通过' if row['speed_pass'] else '未通过'} |\n"
    text+='\n原评估器还要求每个环境拥有超过50个稳态采样。早期失稳会使该检查失败，即使该组总体完成率仍≥95%；不能剔除早期失败环境后重新计算来判通过。\n'
    text+=f'''
## 证据与复现

- [第一段完整记录](experiments/{a.first.name}/validation_report.json)
- [最终完整记录](experiments/{a.final.name}/validation_report.json)
- [最终阶段预先声明的计划](experiments/{a.final.name}/plan.json)
- [训练时间与更新次数](experiments/{a.final.name}/training_audit.json)
- [比较图](experiments/{a.final.name}/validation_comparison.png)
- 每组原始结果和轨迹保存在各实验目录的 `results/`；远端训练日志、优化器和AMP状态保留在 `~/honor_ws/running_sim_pipeline/logs/` 下的同名实验目录。
- 训练入口新增 `--transition-focus --fixed-learning-rate`，第二段另加 `--transition-preserve-speed`，保留教师时另加 `--retention-teacher`。准确命令在各 `training_audit.json`。

## 适用范围

本轮只有一个训练种子，未做AMP消融；各种子环境共享合成地形，完成率不等于户外可靠性。没有验证同一次崎岖地形600秒持续跑、真实雷达/里程计、实机关节标定、电池续航或电机温升。新的候选也没有重新运行导航演示，原导航录像与原基线哈希保持对应。
'''
    a.output.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
