"""Sequential simulation curriculum; each subprocess is logged and checked."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from run_config import ROOT,REPO

PY=REPO/'.venv/bin/python'


def run(args,log):
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','MUJOCO_GL':'egl'}
    with (ROOT/log).open('w') as out:
        subprocess.run([str(PY),*[str(x) for x in args]],cwd=REPO,env=environment,stdout=out,stderr=subprocess.STDOUT,check=True)


def newest(folder):
    return max(folder.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))


if __name__=='__main__':
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--after-robust-checkpoint',type=Path)
    args=parser.parse_args()
    flat=args.after_robust_checkpoint or ROOT/'logs/flat_v2_seed42/model_799.pt'
    while True:
        if flat.exists():
            try:
                checkpoint=torch.load(flat,weights_only=False,map_location='cpu')
                if 'amp' in checkpoint:break
            except (RuntimeError,EOFError,OSError):pass
        time.sleep(2.)
    stages=[('rough',1400)] if args.after_robust_checkpoint else [('robust',800),('rough',1400)]
    for stage,iterations in stages:
        folder=ROOT/'logs'/f'{stage}_seed42'
        (ROOT/'curriculum_status.json').write_text(json.dumps(dict(stage=stage,status='training',parent_pid=os.getpid(),resume=str(flat),iterations=iterations),indent=2))
        run([ROOT/'train.py','--stage',stage,'--num-envs','1024','--iterations',str(iterations),
             '--save-interval','200','--resume',flat,'--log-dir',folder]+(['--reset-amp'] if stage=='rough' else []),f'train_{stage}.log')
        flat=newest(folder)
        for condition,scenario,duration in [('flat','transitions',40),('robust','transitions',40),('rough','straight',25)]:
            run([ROOT/'evaluate.py','--checkpoint',flat,'--stage',condition,'--scenario',scenario,
                 '--speed','2.4','--duration',str(duration),'--num-envs','16','--output',ROOT/'results'/f'{stage}_final_{condition}.json'],f'eval_{stage}_final_{condition}.log')
    (ROOT/'curriculum_status.json').write_text(json.dumps(dict(stage='rough',status='trained_and_initially_evaluated',checkpoint=str(flat)),indent=2))
