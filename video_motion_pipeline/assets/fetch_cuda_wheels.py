from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,time,hashlib
root=Path(__file__).resolve().parent; (root/'wheels').mkdir(exist_ok=True)
urls=[l.split(' @ ',1)[1].strip() for l in (root/'requirements_fast.txt').read_text().splitlines() if l.startswith('nvidia-')]
def fetch(url):
 clean,expected=url.split('#sha256=');name=clean.rsplit('/',1)[-1];p=root/'wheels'/name
 print('FETCH',name,flush=True);t=time.time();last=t;size=0;h=hashlib.sha256()
 with urllib.request.urlopen(clean,timeout=60) as r,open(str(p)+'.part','wb') as f:
  while True:
   b=r.read(4*1024*1024)
   if not b:break
   f.write(b);h.update(b);size+=len(b)
   if time.time()-last>10:print(name,round(size/1e6),'MB',flush=True);last=time.time()
 assert h.hexdigest()==expected,(name,h.hexdigest(),expected)
 Path(str(p)+'.part').rename(p);print('WHEEL_VERIFIED',name,size,round(time.time()-t,1),flush=True)
with ThreadPoolExecutor(2) as pool:list(pool.map(fetch,urls))
