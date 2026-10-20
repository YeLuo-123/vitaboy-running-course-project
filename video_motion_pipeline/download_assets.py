from pathlib import Path
import urllib.request,time,json
from concurrent.futures import ThreadPoolExecutor
root=Path('/home/robotit/honor_ws/GVHMR/inputs/checkpoints')
files={
'gvhmr/gvhmr_siga24_release.ckpt':'camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt',
'hmr2/epoch=10-step=25000.ckpt':'camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt',
'vitpose/vitpose-h-multi-coco.pth':'camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth',
'yolo/yolov8x.pt':'camenduru/GVHMR/resolve/main/yolo/yolov8x.pt',
'body_models/smpl/SMPL_NEUTRAL.pkl':'camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl',
'body_models/smplx/SMPLX_NEUTRAL.npz':'camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz'}
def get(item):
 name,path=item;out=root/name;out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists() and out.stat().st_size>10000:return name,'existing',out.stat().st_size
 for attempt in range(4):
  try:
   url='https://hf-mirror.com/'+path+'?download=true'
   with urllib.request.urlopen(url,timeout=120) as r,open(str(out)+'.part','wb') as f:
    size=0;t=time.time()
    while True:
     b=r.read(4*1024*1024)
     if not b:break
     f.write(b);size+=len(b)
     if time.time()-t>30:print(name,round(size/1e6),'MB',flush=True);t=time.time()
   if size<10000:raise ValueError('too small')
   Path(str(out)+'.part').rename(out);print('DONE',name,size,flush=True)
   return name,'downloaded',size
  except Exception as e:print('RETRY',name,repr(e),flush=True);time.sleep(2)
 raise RuntimeError(name)
with ThreadPoolExecutor(max_workers=4) as p: result=list(p.map(get,files.items()))
Path('assets/download_manifest.json').write_text(json.dumps(dict(source_instructions='https://huggingface.co/spaces/LittleFrog/GVHMR/blob/main/app.py',transport_mirror='https://hf-mirror.com',files=files,results=result),indent=2))
