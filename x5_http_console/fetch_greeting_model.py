#!/usr/bin/env python3
"""Download the official pinned Chinese Vosk model for offline greetings."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip'
SHA256 = '3af8b0e7e0f835ae9d414ce5df580237a3cfb08d586c9fbbb0f7ff29ad5b14ba'


def main():
    root = Path(__file__).resolve().parent/'models'; root.mkdir(exist_ok=True)
    archive = root/'vosk-model-small-cn-0.22.zip'
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        temporary = archive.with_suffix('.zip.part')
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as stream:
            while data := response.read(1024*1024): stream.write(data)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('Vosk model SHA-256 mismatch')
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.namelist():
            target = (root/member).resolve()
            if not target.is_relative_to(root.resolve()): raise ValueError('Unsafe archive member')
        source.extractall(root)
    print('Verified Chinese model:', root/'vosk-model-small-cn-0.22')


if __name__ == '__main__': main()
