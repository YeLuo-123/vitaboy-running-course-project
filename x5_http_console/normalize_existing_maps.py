#!/usr/bin/env python3
"""Convert previously saved raw maps to the 0/100/-1 values required by Nav2 AMCL."""
import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import yaml

from core import amcl_map_pixels, valid_name

ROOT = Path(__file__).resolve().parent


def read_pgm(path):
    with path.open('rb') as file:
        if file.readline().strip() != b'P5':
            raise ValueError(f'{path}: only binary P5 PGM is supported')
        def line():
            value = file.readline().strip()
            while value.startswith(b'#'):
                value = file.readline().strip()
            return value
        width, height = map(int, line().split())
        if line() != b'255':
            raise ValueError(f'{path}: expected 8-bit PGM')
        raw = file.read()
    if len(raw) != width * height:
        raise ValueError(f'{path}: pixel length does not match dimensions')
    return width, height, np.frombuffer(raw, dtype=np.uint8)


def normalize(directory, backup_root, dry_run=False):
    valid_name(directory.name)
    config = yaml.safe_load((directory/'map.yaml').read_text(encoding='utf-8'))
    if config.get('mode') != 'raw' or config.get('image') != 'map.pgm':
        return None
    path = directory/'map.pgm'
    width, height, raw = read_pgm(path)
    if np.any((raw > 100) & (raw != 255)):
        raise ValueError(f'{directory.name}: unexpected raw occupancy value')
    occupancy = np.where(raw == 255, -1, raw).astype(np.int16)
    pixels = amcl_map_pixels(occupancy)
    before = {'free':int(np.count_nonzero(raw == 0)),
              'occupied':int(np.count_nonzero(raw == 100)),
              'unknown':int(np.count_nonzero(raw == 255))}
    after = {'free':int(np.count_nonzero(pixels == 0)),
             'occupied':int(np.count_nonzero(pixels == 100)),
             'unknown':int(np.count_nonzero(pixels == 255))}
    result = {'map':directory.name,'size':[width,height],'before':before,'after':after,
              'changed':not np.array_equal(raw,pixels)}
    if dry_run or not result['changed']:
        return result
    backup = backup_root/directory.name
    if backup.exists():
        raise FileExistsError(f'backup already exists: {backup}')
    backup.mkdir(parents=True)
    shutil.copy2(directory/'map.yaml', backup/'map.yaml')
    shutil.copy2(path, backup/'map.pgm')
    temp = directory/'.map.pgm.normalizing'
    try:
        with temp.open('xb') as file:
            file.write(f'P5\n{width} {height}\n255\n'.encode())
            file.write(pixels.tobytes())
            file.flush(); os.fsync(file.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--backup',type=Path,default=ROOT/'backups'/'20261028_amcl_map')
    parser.add_argument('names',nargs='*',help='map names; default: all saved maps')
    args = parser.parse_args()
    names = args.names or [path.parent.name for path in sorted((ROOT/'maps').glob('*/map.yaml'))]
    for name in names:
        result = normalize(ROOT/'maps'/valid_name(name),args.backup,args.dry_run)
        if result: print(result)
