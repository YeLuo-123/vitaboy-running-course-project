"""Start compilation and offline inference only; no control process is started."""
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
assert root.parent == Path('/data/hihonor_onnx_keyboard')
(root / 'logs').mkdir(exist_ok=True)
with (root / 'logs/build.log').open('ab') as output:
    proc = subprocess.Popen(['bash', str(root / 'onboard/build_onboard.sh')], stdin=subprocess.DEVNULL,
        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
(root / 'logs/build.pid').write_text(str(proc.pid))
print('Compilation/offline-only process PID', proc.pid)
