"""
teleop_franka.py — Franka FR3 arm teleoperation via Jacobian IK + retargeting.

Pipeline (30 Hz vision loop):
  ZED left camera frame (monocular MediaPipe Pose)
    → pose_world_landmarks  (metric 3-D, hip-centred)
    → wrist 3-D position relative to shoulder
    → OneEuroFilter on the 3-D target position   ← filter in Cartesian space
    → FrankaJacobianIK.solve()                   ← DLS IK + null-space
    → MuJoCo position actuators

Why IK instead of direct angle retargeting?
  • Workspace control: your wrist moves → Franka EE follows in 3-D space.
  • All 7 DOF are used — no locked joints.
  • Null-space attraction keeps the elbow in a natural upright posture.
  • Filtering happens in Cartesian space, so the EE trajectory is smooth.

Run with mjpython (NOT plain python — cv2.imshow conflicts with Cocoa on macOS):
    mjpython teleop_franka.py

Tuning guide:
    WORKSPACE_SCALE  — how much 1 m of human arm motion maps to robot motion
    POS_MC / POS_BETA — target-position filter aggressiveness
    FrankaJacobianIK.DAMPING   — DLS regularisation (singularity safety)
    FrankaJacobianIK.NULL_GAIN — strength of elbow-posture attraction
    FrankaJacobianIK.NEUTRAL_MP — your "ready" arm vector → home robot pose
    FrankaJacobianIK.DEBUG = True — prints IK error + joint values per frame
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from vision.camera              import ZEDCamera
from vision.detectors           import ArmTracker
from vision.smoother            import OneEuroFilter
from robots.franka.ik_retargeting import (
    FrankaJacobianIK, HOME, SHOULDER_R, WRIST_R, _lm3
)

# ── Tunable constants ──────────────────────────────────────────────────────────
CAMERA_ID    = 0        # ZED device index (0 = first detected camera)
TARGET_ARM   = "right"  # "right" or "left" (pick the arm you want to mirror)

# One Euro Filter on the 3-D EE target position
# Filtering in Cartesian space = smoother EE trajectory, no joint-space artefacts
POS_FREQ  = 30.0   # Hz: expected vision loop rate
POS_MC    = 0.8    # min_cutoff: lower → smoother at rest
POS_BETA  = 0.005  # beta:       higher → less lag during fast motion

# Visibility threshold — landmarks below this confidence are ignored
VIS_MIN   = 0.5

# Camera display (disabled by default — cv2.imshow conflicts with mjpython Cocoa)
SHOW_CAMERA = False

# ── Paths ──────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "robots", "franka", "scene.xml")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _init_arm(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset Franka to its home pose before the simulation loop starts."""
    mujoco.mj_resetDataKeyframe(model, data, 0)   # keyframe "home"
    mujoco.mj_forward(model, data)


def _extract_wrist(pose_result, vis_min: float = VIS_MIN):
    """
    Return (wrist_world, shoulder_world) as (3,) arrays in MediaPipe world frame,
    or (None, None) if either landmark is not sufficiently visible.
    """
    if pose_result is None or pose_result.pose_world_landmarks is None:
        return None, None

    lm = pose_result.pose_world_landmarks.landmark

    # Select landmark indices based on which arm we track
    sh_idx = SHOULDER_R if TARGET_ARM == "right" else 11   # SHOULDER_L = 11
    wr_idx = WRIST_R    if TARGET_ARM == "right" else 15   # WRIST_L    = 15

    if lm[sh_idx].visibility < vis_min or lm[wr_idx].visibility < vis_min:
        return None, None

    return _lm3(lm, wr_idx), _lm3(lm, sh_idx)


def _update(data:    mujoco.MjData,
            zed:     ZEDCamera,
            tracker: ArmTracker,
            ik:      FrankaJacobianIK,
            pos_f:   OneEuroFilter) -> None:
    """Single-frame update: capture → detect → IK → actuate."""
    frame_l, _ = zed.get_frames()
    if frame_l is None:
        return

    result = tracker.process(frame_l)

    wrist, shoulder = _extract_wrist(result)

    if wrist is None:
        # Pose lost — hold last IK solution (arm stays in place)
        if SHOW_CAMERA:
            _draw_status(frame_l, "No pose — holding")
        _show(frame_l)
        return

    # Filter the 3-D wrist position in Cartesian space
    wrist_smooth = pos_f(wrist)

    # Solve IK: human wrist → robot EE target → joint angles
    q = ik.solve(wrist_smooth, shoulder)

    # Drive MuJoCo position actuators
    data.ctrl[:] = q

    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, result)
        _draw_status(frame_l, f"Franka IK | arm={TARGET_ARM}")

    _show(frame_l)


def _draw_status(frame, text: str) -> None:
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)


def _show(frame) -> None:
    if SHOW_CAMERA:
        cv2.imshow("ZED Left — Franka IK Teleop", frame)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    # Hardware
    zed     = ZEDCamera(camera_id=CAMERA_ID, y_offset=0)
    tracker = ArmTracker()

    # Physics
    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data  = mujoco.MjData(model)

    # IK solver — pass only model; it creates its own internal MjData for FK
    ik    = FrankaJacobianIK(model)

    # Cartesian position filter (3-D)
    pos_f = OneEuroFilter(POS_FREQ, min_cutoff=POS_MC, beta=POS_BETA)

    # Start at home
    _init_arm(model, data)

    print("─" * 60)
    print("  Franka FR3 — Jacobian IK Teleoperation")
    print(f"  Tracking your {TARGET_ARM} arm.")
    print("  Hold arm slightly forward at shoulder height to rest at home.")
    print("  Move wrist in 3-D → Franka EE follows.")
    print("  Press ESC in the MuJoCo viewer to quit.")
    print("─" * 60)
    print(f"  IK: DLS λ={ik.DAMPING}  null_gain={ik.NULL_GAIN}  iters={ik.N_ITERS}")
    print(f"  Home EE pos: {ik.home_ee_pos.round(3)} m")
    print("─" * 60)

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            _update(data, zed, tracker, ik, pos_f)
            mujoco.mj_step(model, data)
            v.sync()

            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    tracker.close()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
