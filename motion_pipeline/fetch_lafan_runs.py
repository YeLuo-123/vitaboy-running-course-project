"""Fetch only run BVHs from the publisher's ZIP using verified byte ranges."""
import io
import json
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip'


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        response = self.request('bytes=-65536')
        header = response.headers['Content-Range']
        self.size = int(header.split('/')[-1])
        self.tail_start = int(header.split()[1].split('-')[0])
        self.tail = response.read()

    def request(self, byte_range):
        response = urllib.request.urlopen(urllib.request.Request(self.url,
            headers={'Range': byte_range, 'Accept-Encoding': 'identity'}), timeout=120)
        if response.status != 206:
            response.close()
            raise RuntimeError('Publisher did not honor byte range')
        return response

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos
    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.pos+size)
        if end <= self.pos: return b''
        if self.pos >= self.tail_start:
            result = self.tail[self.pos-self.tail_start:end-self.tail_start]
        else:
            response = self.request(f'bytes={self.pos}-{end-1}')
            expected = f'bytes {self.pos}-{end-1}/{self.size}'
            if response.headers['Content-Range'] != expected:
                raise RuntimeError(f'Range mismatch: {response.headers["Content-Range"]} != {expected}')
            result = response.read()
        self.pos = end
        return result


root = Path(__file__).resolve().parent / 'data/raw'
root.mkdir(parents=True, exist_ok=True)
records = []
with zipfile.ZipFile(HttpRangeFile(URL)) as archive:
    names = [n for n in archive.namelist() if Path(n).name.startswith('run') and n.endswith('.bvh')]
    print('RUN FILES', names, flush=True)
    for name in names:
        out = root / Path(name).name
        if not out.exists():
            out.write_bytes(archive.read(name))
        records.append(dict(file=out.name, archive_path=name, bytes=out.stat().st_size,
            sha256=hashlib.sha256(out.read_bytes()).hexdigest(), source=URL))
        print('DOWNLOADED', out.name, out.stat().st_size, flush=True)
for name in ('license.txt', 'README.md'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-animation-dataset/master/'+name,
                               root / ('LAFAN1_'+name))
(root/'source_manifest.json').write_text(json.dumps(records, indent=2))
