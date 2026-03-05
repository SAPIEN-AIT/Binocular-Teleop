"""
teleop_franka.py — Franka FR3 arm teleoperation via MediaPipe Pose.

Pipeline (30 Hz vision loop):
  ZED left camera frame (monocular)
    → MediaPipe Pose arm tracking
    → direct angle retargeting  (arm geometry → 7 Franka joint angles)
    → One Euro filtered joint targets
    → MuJoCo position actuators

No mocap body or wrist orientation needed — Franka is a fixed-base arm
driven purely by joint position setpoints.

Run with mjpython (NOT plain python — cv2.imshow conflicts with Cocoa on macOS):
    mjpython teleop_franka.py

Tuning guide:
    JOINT_MC / JOINT_BETA  — joint filter aggressiveness
    TARGET_ARM             — "right" or "left" to select which arm to mirror
    FrankaIKRetargeter.DEBUG_ANGLES = True  — print live joint values
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from vision.camera                      import ZEDCamera
from vision.detectors                   import ArmTracker
from vision.smoother                    import OneEuroFilter
from robots.franka.ik_retargeting       import FrankaIKRetargeter, HOME

# ── Tunable constants ─────────────────────────────────────────────────────────
CAMERA_ID    = 0           # ZED device index

# Which arm to track.  "right" = user's right arm (appears on left of image
# on a non-mirrored ZED).  Flip to "left" if you prefer the other arm.
TARGET_ARM   = "right"

# One Euro Filter — joints (7-dim)
JOINT_FREQ   = 30.0        # Hz: expected vision loop rate
JOINT_MC     = 1.0         # min_cutoff: lower → smoother at rest
JOINT_BETA   = 0.05        # beta:       higher → less lag during fast motion

# Camera display
SHOW_CAMERA  = False       # cv2.imshow conflicts with mjpython Cocoa on macOS

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "robots", "franka", "scene.xml")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _init_arm(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset Franka to its home pose before the simulation loop starts."""
    mujoco.mj_resetDataKeyframe(model, data, 0)   # keyframe "home"
    mujoco.mj_forward(model, data)


def _update(data:    mujoco.MjData,
            zed:     ZEDCamera,
            tracker: ArmTracker,
            ik:      FrankaIKRetargeter,
            joint_f: OneEuroFilter) -> None:
    """Single-frame update: capture → detect → retarget → actuate."""
    frame_l, _ = zed.get_frames()
    if frame_l is None:
        return

    result = tracker.process(frame_l)

    if result.pose_landmarks is None:
        if SHOW_CAMERA:
            cv2.putText(frame_l, "No pose", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        _show(frame_l)
        return

    # ── Retarget ──────────────────────────────────────────────────────────
    q_raw    = ik.retarget(result)
    q_smooth = joint_f(q_raw)
    data.ctrl[:] = q_smooth

    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, result)
        cv2.putText(frame_l, "Franka Teleop", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    _show(frame_l)


def _show(frame):
    if SHOW_CAMERA:
        cv2.imshow("ZED Left — Franka Teleop", frame)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Hardware
    zed     = ZEDCamera(camera_id=CAMERA_ID, y_offset=0)
    tracker = ArmTracker()

    # Physics
    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data  = mujoco.MjData(model)

    # Retargeter and smoother
    ik      = FrankaIKRetargeter(model)
    joint_f = OneEuroFilter(JOINT_FREQ, min_cutoff=JOINT_MC, beta=JOINT_BETA)

    # Start at home pose
    _init_arm(model, data)

    print("─" * 60)
    print("  Franka FR3 Arm Teleoperation (MediaPipe Pose)")
    print(f"  Tracking your {TARGET_ARM} arm.")
    print("  Stand with your arm visible to the ZED camera.")
    print("  Press  ESC  in the MuJoCo viewer to quit.")
    print("─" * 60)

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            _update(data, zed, tracker, ik, joint_f)

            mujoco.mj_step(model, data)
            v.sync()

            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    tracker.close()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
