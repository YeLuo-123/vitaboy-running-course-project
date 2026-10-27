"""Map-bound room polygons and their single navigation target."""
import json
import math
import os
from pathlib import Path
import secrets

import numpy as np

from core import FREE_LIMIT, world_to_cell

MAX_AREAS = 60
MAX_VERTICES = 128


def area_name(value):
    if not isinstance(value, str):
        raise ValueError('区域名称必须是文字')
    name = value.strip()
    if not 1 <= len(name) <= 24 or any(ord(ch) < 32 for ch in name):
        raise ValueError('区域名称需为 1–24 个可见字符')
    return name


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('区域顶点须为 [x, y]')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError('区域坐标必须是有限数值')
    return [float(value[0]), float(value[1])]


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _on_segment(a, b, p):
    return abs(_cross(a,b,p)) < 1e-9 and min(a[0],b[0])-1e-9 <= p[0] <= max(a[0],b[0])+1e-9 and min(a[1],b[1])-1e-9 <= p[1] <= max(a[1],b[1])+1e-9


def _intersects(a,b,c,d):
    if _on_segment(a,b,c) or _on_segment(a,b,d) or _on_segment(c,d,a) or _on_segment(c,d,b):
        return True
    return _cross(a,b,c)*_cross(a,b,d) < 0 and _cross(c,d,a)*_cross(c,d,b) < 0


def point_in_polygon(point, polygon):
    x,y = point
    inside = False
    for i,a in enumerate(polygon):
        b = polygon[(i+1)%len(polygon)]
        if _on_segment(a,b,point): return True
        if (a[1] > y) != (b[1] > y):
            crossing = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if x < crossing: inside = not inside
    return inside


def validate_polygon(value, info):
    if not isinstance(value, list) or not 3 <= len(value) <= MAX_VERTICES:
        raise ValueError('区域轮廓需要 3–128 个顶点')
    points = [_point(p) for p in value]
    if any(not (0 <= world_to_cell(*p,info)[0] < info['width'] and
                0 <= world_to_cell(*p,info)[1] < info['height']) for p in points):
        raise ValueError('区域顶点必须位于当前地图内')
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2
    if area < .04: raise ValueError('区域轮廓面积太小')
    n=len(points)
    for i in range(n):
        for j in range(i+1,n):
            if j in ((i+1)%n,(i-1)%n): continue
            if _intersects(points[i],points[(i+1)%n],points[j],points[(j+1)%n]):
                raise ValueError('区域轮廓不能交叉')
    return points


def validate_target(value, polygon, info, grid):
    if not isinstance(value, dict): raise ValueError('目标点格式无效')
    x,y = _point([value.get('x'),value.get('y')])
    heading = value.get('yaw',0)
    if type(heading) not in (int,float) or not math.isfinite(heading):
        raise ValueError('目标朝向必须是有限数值')
    if not point_in_polygon((x,y),polygon): raise ValueError('目标点必须在所属区域内')
    col,row = world_to_cell(x,y,info)
    if not (0 <= col < info['width'] and 0 <= row < info['height']):
        raise ValueError('目标点超出地图')
    cells = np.asarray(grid,dtype=np.int8).reshape(info['height'],info['width'])
    if not 0 <= cells[row,col] <= FREE_LIMIT:
        raise ValueError('目标点必须位于已知空闲区域')
    radius=max(1,math.ceil(.15/info['resolution']))
    y0,y1=max(0,row-radius),min(info['height'],row+radius+1)
    x0,x1=max(0,col-radius),min(info['width'],col+radius+1)
    around=cells[y0:y1,x0:x1]
    if np.any((around < 0)|(around >= 65)):
        raise ValueError('目标点距离未知区域或障碍物过近')
    return {'x':x,'y':y,'yaw':float(heading)}


def load_areas(map_dir):
    path=Path(map_dir)/'areas.json'
    if not path.exists(): return []
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or data.get('version') != 1 or not isinstance(data.get('areas'),list):
        raise ValueError('区域标注文件格式不受支持')
    return data['areas']


def _write(map_dir, areas):
    directory=Path(map_dir)
    path=directory/'areas.json'
    tmp=directory/('.areas-'+secrets.token_hex(6)+'.tmp')
    try:
        tmp.write_text(json.dumps({'version':1,'areas':areas},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def save_area(map_dir, payload, info, grid):
    name=area_name(payload.get('area_name'))
    polygon=validate_polygon(payload.get('polygon'),info)
    areas=load_areas(map_dir)
    identifier=payload.get('id')
    existing=next((a for a in areas if a['id']==identifier),None) if identifier else None
    if identifier and existing is None: raise ValueError('区域不存在')
    if any(a['name'].casefold()==name.casefold() and a['id']!=identifier for a in areas):
        raise ValueError('区域名称已存在')
    if existing is None and len(areas)>=MAX_AREAS: raise ValueError('区域数量已达上限')
    target=existing.get('target') if existing else None
    if target:
        try: target=validate_target(target,polygon,info,grid)
        except ValueError: target=None
    record={'id':identifier if existing else secrets.token_hex(8),'name':name,'polygon':polygon,'target':target}
    areas=[record if a['id']==identifier else a for a in areas] if existing else areas+[record]
    _write(map_dir,areas)
    return record


def set_target(map_dir,identifier,target,info,grid):
    areas=load_areas(map_dir)
    area=next((a for a in areas if a['id']==identifier),None)
    if area is None: raise ValueError('区域不存在')
    area['target']=validate_target(target,area['polygon'],info,grid)
    _write(map_dir,areas)
    return area


def delete_area(map_dir,identifier):
    areas=load_areas(map_dir)
    remaining=[a for a in areas if a['id']!=identifier]
    if len(remaining)==len(areas): raise ValueError('区域不存在')
    _write(map_dir,remaining)
    return {'ok':True}
