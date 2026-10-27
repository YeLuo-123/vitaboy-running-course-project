"""Collect versioned evidence and candidate artifacts without replacing baseline."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
RUNS=[ROOT/'experiments'/f'transition_20261025_v{i}' for i in [1,2,3]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_readme(root, report):
    path=root/'README.md'
    text=path.read_text(encoding='utf-8')
    heading='## 专项优化记录（2026-10-25）'
    if heading in text:raise RuntimeError('README already contains this experiment')
    outcome='通过本轮全部检查，新候选单独保存在实验目录。' if report['all_checks_passed'] else '起停转向得到改善，但未通过全部保留检查；默认交付模型保持原版。'
    intro=f'''{heading}

本轮追加1400次正式更新，测试了起停课程、速度/停稳奖励与训练期直跑保留约束。{outcome}
结果与失败记录见 [专项优化报告](TRANSITION_OPTIMIZATION_REPORT.md) 和 [最终验证JSON](experiments/transition_20261025_v3/validation_report.json)。
候选ONNX：`experiments/transition_20261025_v3/candidate/policy.onnx`；原 `artifacts/running_policy.onnx` 不变。
Tracking动态AMP专家仍为原数据；新增的直跑保留教师是原交付跑步actor，仅用于训练，部署时不带教师。

'''
    first,rest=text.split('\n',1)
    path.write_text(first+'\n\n'+intro+rest.lstrip('\n'),encoding='utf-8')


def main():
    final=RUNS[-1]
    report=json.loads((final/'validation_report.json').read_text())
    plan=json.loads((final/'plan.json').read_text())
    assert sha(ROOT/'artifacts/running_policy.onnx')==plan['baseline_policy_sha256']
    for name in ['run_config.py','amp.py','evaluate.py','evaluate_cpu.py']:
        assert sha(ROOT/name)==sha(RUNS[0]/'source_before'/name),name
    for run in RUNS:
        log=ROOT/'logs'/run.name
        config=ROOT/'configs'/run.name
        config.mkdir(exist_ok=False)
        for name in ['agent.yaml','env.yaml','manifest.json']:
            shutil.copy2(log/name,config/name)
        shutil.copytree(log/'code',config/'code')
        for name in ['effective_config_audit.json','training_audit.json','plan.json']:
            if (run/name).exists():shutil.copy2(run/name,config/name)
    tests=subprocess.run([sys.executable,str(ROOT/'test_transition_optimization.py')],capture_output=True,text=True)
    (final/'contract_tests.txt').write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Contract tests failed')
    for run in [RUNS[0],final]:
        subprocess.run([sys.executable,str(ROOT/'plot_transition_experiment.py'),'--run',str(run)],check=True)
    subprocess.run([sys.executable,str(ROOT/'write_transition_report.py'),'--first',str(RUNS[0]),
                    '--middle',str(RUNS[1]),'--final',str(final),'--output',str(ROOT/'TRANSITION_OPTIMIZATION_REPORT.md')],check=True)
    update_readme(ROOT,report)
    active=ROOT/'active_run.json'
    if active.exists():
        value=json.loads(active.read_text())
        if Path(value.get('log_dir','')).name==final.name:
            value.update(running=False,completed_updates=200,experiment_complete=True)
            active.write_text(json.dumps(value,indent=2))
    provenance=dict(experiment=final.name,accepted=report['all_checks_passed'],baseline_replaced=False,
        candidate_policy_sha256=sha(final/'candidate/policy.onnx'),
        candidate_checkpoint_sha256=sha(final/'candidate/policy.pt'),
        baseline_policy_sha256=sha(ROOT/'artifacts/running_policy.onnx'),
        evaluation_source_unchanged=True,contract_tests_passed=True,
        hardware_execution=False,formal_updates=1400,
        source_sha256={name:sha(ROOT/name) for name in ['train.py','run_config.py','amp.py','evaluate.py',
            'evaluate_cpu.py','transition_optimization.py','retention_prior.py','validate_transition_candidate.py']})
    (final/'provenance.json').write_text(json.dumps(provenance,indent=2))
    # Curated archive: full numeric evidence and actual source snapshots; no
    # duplicate baseline weights, smoke checkpoints or rejected actor weights.
    files=set()
    for run in RUNS:
        files.update(run.glob('*.json'))
        files.update(run.glob('*.png'))
        files.update(run.glob('*.txt'))
        files.update(run.glob('*.log'))
        files.update((run/'results').glob('*.json'))
        files.update((ROOT/'configs'/run.name).rglob('*'))
    files.update((final/'candidate').glob('*'))
    files.add(ROOT/'TRANSITION_OPTIMIZATION_REPORT.md')
    # Include baseline acceptance records, not the copied baseline models.
    for run in RUNS:files.add(run/'baseline/simulation_acceptance.json')
    files=sorted(f for f in files if f.is_file())
    receipt=dict(files={str(f.relative_to(ROOT)):sha(f) for f in files},provenance=provenance)
    receipt_path=final/'delivery_manifest.json'
    receipt_path.write_text(json.dumps(receipt,indent=2))
    files.append(receipt_path)
    archive=ROOT/'experiments/transition_20261025_delivery.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                          accepted=report['all_checks_passed']),indent=2))


if __name__=='__main__':
    main()
