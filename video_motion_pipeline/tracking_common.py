"""Use the verified Tracking setup with this video's separate motion file."""
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parent
MOTION=ROOT/'outputs/vita_boy/video_motion.npz'
spec=importlib.util.spec_from_file_location('_run_tracking_config',ROOT.parent/'motion_pipeline/tracking_common.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.MOTION=MOTION


def configure_tracking(cfg, *, training=False, jitter=False):
    cfg=base.configure_tracking(cfg,training=training,jitter=jitter)
    cfg.viewer.azimuth=0.
    return cfg
