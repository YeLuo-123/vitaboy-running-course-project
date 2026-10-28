#!/usr/bin/env python3
"""Download pinned YOLO-World v2/CLIP artifacts and verify their SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'config/vision_models.json').read_text())
    directory = root/'models'; directory.mkdir(exist_ok=True)
    for item in manifest['files']:
        target = directory/item['file']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
            print('Verified', target.name); continue
        url = f"https://huggingface.co/{item['repository']}/resolve/{item['revision']}/{item['source']}"
        temporary = target.with_suffix(target.suffix+'.part')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024*1024): stream.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('SHA-256 mismatch: '+target.name)
        temporary.replace(target); print('Downloaded and verified', target.name)


if __name__ == '__main__': main()
