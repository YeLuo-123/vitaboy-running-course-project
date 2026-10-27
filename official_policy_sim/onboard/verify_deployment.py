"""Verify approved deployment artifact hashes without loading the controller."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / 'verified_artifacts.json').read_text())
for name, expected in manifest['sha256'].items():
    path = (root / name).resolve()
    if root not in path.parents:
        raise SystemExit('Invalid artifact path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Artifact changed; revalidate before starting: ' + name)
print('Deployment hashes verified. This is not a hardware movement test.')
