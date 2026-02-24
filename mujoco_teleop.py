"""
mujoco_teleop.py — Hand teleoperation with direct angle retargeting.

Pipeline (30 Hz vision loop):
  ZED left camera frame (monocular mode)
    → MediaPipe hand tracking (single camera)
    → direct angle retargeting  (MediaPipe 3-D joint angles → 16 LEAP joints)
    → One Euro filtered mocap position  (hand proxy follows wrist, Y fixed)
    → One Euro filtered joint targets
    → MuJoCo position actuators

  Stereo depth is intentionally disabled (STEREO_DEPTH = False).
  Once IK is fully tuned, flip that flag to re-enable epipolar + triangulation.

Run with mjpython (NOT plain python — cv2.imshow conflicts with Cocoa on macOS):
    mjpython mujoco_teleop.py

Tuning guide (constants block below):
    JOINT_MC / JOINT_BETA  — joint filter: lower MC = smoother, higher beta = less lag
    POS_MC   / POS_BETA    — position filter
    X/Z_SCALE              — workspace size in simulation metres
    DEPTH_SCALE            — how much stereo depth maps to sim-Y movement (only when STEREO_DEPTH=True)
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from binocular.camera        import ZEDCamera
from binocular.detectors     import StereoHandTracker
import binocular.geometry    as geo
from binocular.smoother      import OneEuroFilter
from binocular.ik_retargeting import IKRetargeter

# ── Tunable constants ─────────────────────────────────────────────────────────
CAMERA_ID    = 0       # ZED device index (try 0 if 1 fails)

# ── Mode toggle ───────────────────────────────────────────────────────────────
# False = monocular (left frame only, Y fixed) → tune IK first.
# True  = stereo    (epipolar + triangulated depth) → re-enable once IK works.
STEREO_DEPTH = False

# One Euro Filter — joints (16-dim)
JOINT_FREQ   = 30.0    # Hz: expected vision loop rate
JOINT_MC     = 1.0     # min_cutoff: lower → smoother at rest
JOINT_BETA   = 0.03    # beta:       higher → less lag during fast motion

# One Euro Filter — wrist position (3-dim)
POS_FREQ     = 30.0
POS_MC       = 0.8
POS_BETA     = 0.005

# Workspace mapping — lateral/vertical
# Image coords are normalised [0,1]; scale controls how far the sim hand moves.
X_SCALE      = 0.3     # lateral range  (m) — sim x
Z_SCALE      = 0.3     # vertical range (m) — sim z

# Depth (stereo Z) → sim Y — only used when STEREO_DEPTH = True
DEPTH_MIN_M  = 0.20
DEPTH_MAX_M  = 0.90
DEPTH_MID_M  = 0.45    # neutral depth → hand sits at START_Y
DEPTH_SCALE  = 0.4     # m of sim-Y movement per m of depth change
START_Y      = 0.30    # initial sim Y (forward) of the hand proxy — also used as fixed Y
START_Z      = 0.30    # initial sim Z (height) of the hand proxy

# Epipolar constraint — only checked when STEREO_DEPTH = True
EPIPOLAR_TOL = 5       # px (tight, since Y_OFFSET_PX corrects the 30 px bias)

# cv2.imshow conflicts with mjpython's Cocoa event loop on macOS.
# Set to True only when running with plain `python` (not `mjpython`).
SHOW_CAMERA  = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "leap_hand", "scene_binocular.xml")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _init_hand(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Teleport the LEAP palm to the starting position before physics runs.

    Without this, the palm starts at its XML-defined origin and falls under
    gravity before the weld constraint can engage on the first frame.
    """
    mid  = model.body("hand_proxy").mocapid[0]
    pos  = np.array([0.0, START_Y, START_Z])
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    data.mocap_pos[mid]  = pos
    data.mocap_quat[mid] = quat

    # Also move the freejoint so palm spawns at the right place
    jid  = model.joint("palm_free").id
    addr = model.jnt_qposadr[jid]
    data.qpos[addr:addr+3] = pos       # position
    data.qpos[addr+3:addr+7] = quat    # quaternion (w, x, y, z)

    mujoco.mj_forward(model, data)


def _update(data:    mujoco.MjData,
            zed:     ZEDCamera,
            tracker: StereoHandTracker,
            ik:      IKRetargeter,
            pos_f:   OneEuroFilter,
            joint_f: OneEuroFilter,
            mid:     int) -> None:
    """
    Single-frame update: capture → detect → retarget → actuate.

    When STEREO_DEPTH=False the right frame and epipolar check are skipped
    entirely — only the left camera needs to see a hand.
    """
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        return

    h, w, _ = frame_l.shape

    if STEREO_DEPTH:
        # ── Stereo path — requires both cameras ──────────────────────────────
        res_l, res_r = tracker.process(frame_l, frame_r)

        if not (res_l.multi_hand_landmarks and res_r.multi_hand_landmarks):
            _show(frame_l)
            return

        lm_l = res_l.multi_hand_landmarks[0].landmark
        lm_r = res_r.multi_hand_landmarks[0].landmark

        # Epipolar check
        py_l = int(lm_l[0].y * h)
        py_r = int(lm_r[0].y * h)
        valid, epi_err = geo.check_epipolar_constraint(py_l, py_r,
                                                        tolerance_px=EPIPOLAR_TOL)
        if not valid:
            if SHOW_CAMERA:
                cv2.putText(frame_l, f"EPIPOLAR REJECTED ({epi_err:.0f}px)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _show(frame_l)
            return

        # Triangulated depth
        px_l = int(lm_l[0].x * w)
        px_r = int(lm_r[0].x * w)
        z_cm = geo.triangulate_depth(px_l, px_r)
        if z_cm is not None and z_cm > 0:
            depth_m = float(np.clip(z_cm / 100.0, DEPTH_MIN_M, DEPTH_MAX_M))
        else:
            depth_m = DEPTH_MID_M

        sim_y = START_Y + (DEPTH_MID_M - depth_m) * DEPTH_SCALE

        if SHOW_CAMERA:
            cv2.putText(frame_l, f"Depth {depth_m*100:.1f} cm  epi={epi_err:.0f}px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    else:
        # ── Monocular path — left frame only ─────────────────────────────────
        res_l, _ = tracker.process(frame_l, frame_r)   # right result ignored

        if not res_l.multi_hand_landmarks:
            if SHOW_CAMERA:
                cv2.putText(frame_l, "No hand", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            _show(frame_l)
            return

        lm_l    = res_l.multi_hand_landmarks[0].landmark
        sim_y   = START_Y   # depth fixed until stereo is re-enabled

        if SHOW_CAMERA:
            cv2.putText(frame_l, "Monocular IK", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    # ── Mocap position ────────────────────────────────────────────────────────
    raw_pos = np.array([
        -(lm_l[0].x - 0.5) * X_SCALE,   # image x → sim x (mirrored)
        sim_y,                            # fixed or triangulated
        START_Z + (0.5 - lm_l[0].y) * Z_SCALE,  # image y → sim z (inverted)
    ])
    data.mocap_pos[mid] = pos_f(raw_pos)

    # ── Direct angle retargeting ──────────────────────────────────────────────
    q_raw    = ik.retarget(None, lm_l)
    q_smooth = joint_f(q_raw)
    data.ctrl[:] = q_smooth

    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, res_l)

    _show(frame_l)


def _show(frame):
    if SHOW_CAMERA:
        cv2.imshow("ZED Left — Binocular Teleop", frame)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Hardware
    # Pass y_offset so vertical alignment is ready when STEREO_DEPTH is re-enabled.
    zed     = ZEDCamera(camera_id=CAMERA_ID,
                        y_offset=geo.Y_OFFSET_PX if STEREO_DEPTH else 0)
    tracker = StereoHandTracker()

    # Physics
    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data  = mujoco.MjData(model)

    # Mocap body index for hand_proxy
    mid = model.body("hand_proxy").mocapid[0]

    # Retargeter and filters
    ik      = IKRetargeter(model)
    pos_f   = OneEuroFilter(POS_FREQ,   min_cutoff=POS_MC,   beta=POS_BETA)
    joint_f = OneEuroFilter(JOINT_FREQ, min_cutoff=JOINT_MC, beta=JOINT_BETA)

    # Spawn hand at rest position
    _init_hand(model, data)

    print("─" * 60)
    print("  Binocular Hand Teleoperation (Direct Angle Retargeting)")
    print("  Move your right hand in front of the ZED camera.")
    print("  Press  Q  in the camera window  or  ESC  in the")
    print("  MuJoCo viewer to quit.")
    print("─" * 60)

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            _update(data, zed, tracker, ik, pos_f, joint_f, mid)

            mujoco.mj_step(model, data)
            v.sync()

            # Allow quitting from the camera window
            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
