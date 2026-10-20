"""Record installed versions and source/asset hashes without environment secrets."""
import hashlib,importlib.metadata,json,platform,subprocess
from pathlib import Path
from run_config import ROOT,REPO
from whole_body_control.assets.robots.vita_boy.vita_boy_constants import VITA_BOY_XML


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    result=subprocess.run(['git','-C',str(REPO),*args],capture_output=True,text=True,check=False)
    return result.stdout.strip() if result.returncode==0 else None


asset_root=Path(VITA_BOY_XML).parent.parent
assets={str(p.relative_to(asset_root)):sha(p) for p in sorted(asset_root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts}
sources={p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
result=dict(python=platform.python_version(),platform=platform.platform(),
    packages=dict(sorted(packages.items())),official_repository=str(REPO),
    official_repository_commit=git('rev-parse','HEAD'),official_repository_status=git('status','--short'),
    official_locomotion_onnx_sha256=sha(REPO/'deploy/checkpoints/loco.onnx'),
    asset_root=str(asset_root),asset_sha256=assets,pipeline_source_sha256=sources,
    note='Per-run code snapshots and YAMLs are in logs/<stage>/; source hashes here describe the delivered pipeline.')
out=ROOT/'artifacts/environment_and_source.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2));print(out)
