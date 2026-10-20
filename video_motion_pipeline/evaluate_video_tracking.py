"""Run the existing finite-clip dynamics test with video-specific criteria.

Running-speed and aerial-gait requirements do not apply to this arm/jump clip.
The original running criteria remain visible separately in the saved result.
"""
from pathlib import Path
import json
import os
import runpy
import sys
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT.parent/'honor_rl_mjlab')
sys.path.insert(0,str(ROOT))
output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
runpy.run_path(str(ROOT.parent/'motion_pipeline/evaluate_tracking.py'),run_name='__main__')
data=json.loads(output.read_text())
data['running_only_acceptance_criteria']=data.pop('engineering_acceptance_criteria')
criteria=dict(completion_at_least_95_percent=data['completed_fraction']>=.95,
              joint_rmse_at_most_0_15_rad=data['joint_rmse_rad']<=.15,
              relative_body_rmse_at_most_0_06_m=data['relative_body_position_rmse_m']<=.06,
              root_rmse_at_most_0_20_m=data['root_position_rmse_m']<=.2,
              no_reference_wrap_or_teleport=data['reference_wraps']==0)
data['engineering_acceptance_criteria']=criteria
data['passes_engineering_acceptance']=all(criteria.values())
data['evaluation_scope']='Video imitation: finite original-timing clip, pose fidelity and no falls; no running-speed target.'
output.write_text(json.dumps(data,indent=2))
print('VIDEO_ACCEPTANCE',json.dumps(criteria),data['passes_engineering_acceptance'])
