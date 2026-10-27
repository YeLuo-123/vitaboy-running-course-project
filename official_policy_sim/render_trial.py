"""Render saved simulation states without running any controller or ROS node."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    path = args.trial.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    data = mujoco.MjData(model)
    states = np.load(path / "trajectory.npz")
    samples, poses = states["samples"], states["qpos"]
    # Skip the explicitly frozen initialization. Rendering only replays states.
    indices = np.flatnonzero(np.diff(samples[:, 1], prepend=0) > .001)[::2]
    camera = mujoco.MjvCamera()
    camera.distance = 3.
    camera.azimuth = -135.
    camera.elevation = -15.
    writer = cv2.VideoWriter(str(path / "keyboard_sim.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25., (960, 540))
    assert writer.isOpened()
    try:
        with mujoco.Renderer(model, height=540, width=960) as renderer:
            for k, i in enumerate(indices):
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                title = f"SIMULATION | Official C++ keyboard policy | t={samples[i,1]:.1f}s"
                cv2.rectangle(image, (0, 0), (960, 70), (25, 25, 25), -1)
                cv2.putText(image, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (245,245,245), 1, cv2.LINE_AA)
                cv2.putText(image, f"vx={samples[i,8]:.2f} m/s   height={samples[i,4]:.2f} m", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, (245,245,245), 1, cv2.LINE_AA)
                writer.write(image)
                if k == min(180, len(indices)-1):
                    cv2.imwrite(str(path / "keyboard_sim.png"), image)
    finally:
        writer.release()
    print(path / "keyboard_sim.mp4", flush=True)


if __name__ == "__main__":
    main()
