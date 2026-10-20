"""Fetch the frozen upstream modules used by this BVH-only GMR pipeline."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import urllib.request

COMMIT='bb1bbe40774794fceb2a7c579a3464a28e68c844'
FILES=['LICENSE','README.md','setup.py',
 'general_motion_retargeting/params.py',
 'general_motion_retargeting/motion_retarget.py',
 'general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json',
 'general_motion_retargeting/utils/lafan1.py',
 'general_motion_retargeting/utils/lafan_vendor/__init__.py',
 'general_motion_retargeting/utils/lafan_vendor/extract.py',
 'general_motion_retargeting/utils/lafan_vendor/utils.py',
 'general_motion_retargeting/utils/lafan_vendor/license.txt']
root=Path(__file__).resolve().parent.parent/'GMR_source'
def fetch(name):
    out=root/name; out.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(f'https://raw.githubusercontent.com/YanjieZe/GMR/{COMMIT}/{name}',out)
    return name
with ThreadPoolExecutor(max_workers=4) as pool:
    for name in pool.map(fetch,FILES): print('FETCH',name,flush=True)
(root/'source_manifest.json').write_text(json.dumps(dict(repository='https://github.com/YanjieZe/GMR',commit=COMMIT,files=FILES),indent=2))
