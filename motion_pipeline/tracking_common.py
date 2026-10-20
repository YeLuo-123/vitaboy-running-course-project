"""Configuration shared by training and independent finite-clip evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOTION = ROOT/'data/retargeted/lafan_run2_s4_geometry/run.npz'


def configure_tracking(cfg, *, training=False, jitter=False):
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.events.clear()
    cfg.observations['actor'].enable_corruption = bool(training or jitter)
    motion = cfg.commands['motion']
    motion.motion_file = str(MOTION)
    motion.debug_vis = False
    motion.pose_range = {'x':(-.01,.01), 'y':(-.01,.01), 'roll':(-.02,.02), 'pitch':(-.02,.02), 'yaw':(-.02,.02)} if training or jitter else {}
    motion.velocity_range = {k:(-.05,.05) for k in ['x','y','z','roll','pitch','yaw']} if training or jitter else {}
    motion.joint_position_range = (-.02,.02) if training or jitter else (0.,0.)
    motion.sampling_mode = 'adaptive' if training else 'start'
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.scene.sensors += (ContactSensorCfg(name='feet_ground_contact',
        primary=ContactMatch(mode='body',pattern=r'^(left_ankle_roll_link|right_ankle_roll_link)$',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'), fields=('found','force'),
        reduce='netforce',track_air_time=True,history_length=3),)
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width, cfg.viewer.height = 640,480
    cfg.viewer.azimuth = 90.
    cfg.viewer.elevation = -12.
    return cfg
