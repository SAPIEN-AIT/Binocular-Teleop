"""
teleop_leap.py — Hand teleoperation with direct angle retargeting.

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
    mjpython teleop_leap.py

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

from vision.camera                       import ZEDCamera
from vision.detectors                    import StereoHandTracker
import vision.geometry                   as geo
from vision.smoother                     import OneEuroFilter
from robots.leap_hand.ik_retargeting     import IKRetargeter, palm_quat

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

# Workspace mapping
# In the new pinhole back-projection model the workspace scales automatically
# with depth (back_project returns real metres).  X_SCALE / Z_SCALE are kept
# here as legacy references but are no longer applied to the position output.
X_SCALE      = 0.3     # legacy — no longer used
Z_SCALE      = 0.3     # legacy — no longer used

# Depth (stereo Z) → sim Y — only used when STEREO_DEPTH = True
DEPTH_MIN_M  = 0.20
DEPTH_MAX_M  = 0.90
DEPTH_MID_M  = 0.45    # neutral depth → hand sits at START_Y
DEPTH_SCALE  = 0.4     # m of sim-Y movement per m of depth change
START_Y      = 0.30    # initial sim Y (forward) of the hand proxy — also used as fixed Y
START_Z      = 0.30    # initial sim Z (height) of the hand proxy

# Epipolar constraint — only checked when STEREO_DEPTH = True
EPIPOLAR_TOL = 5       # px (tight, since Y_OFFSET_PX corrects the 30 px bias)

# ── Wrist orientation ─────────────────────────────────────────────────────────────────────
# Disabled while tuning finger IK — enable once fingers track correctly.
# When True, rotate ORIENT_OFFSET until the sim palm matches your real palm at rest.
WRIST_ORIENT  = False  # drive mocap_quat from palm normal; False = locked identity

# One Euro Filter for quaternion (4-dim)
ORIENT_FREQ   = 30.0
ORIENT_MC     = 1.5    # higher than pos — rotations need faster response
ORIENT_BETA   = 0.05

# Static rotation applied AFTER the MediaPipe-derived quaternion.
# Compensates for the LEAP hand's rest orientation in the XML.
# Tune until the sim palm matches your real palm when flat facing the camera.
# Format: (w, x, y, z).  Identity = no offset.
ORIENT_OFFSET = np.array([0.707, -0.707, 0.0, 0.0])   # Rx(-90°): cancels palm-facing-camera rest pose

# ── Handedness filter ─────────────────────────────────────────────────────────────────────
# ZED is a non-mirrored camera: your RIGHT hand appears on the LEFT side of the
# image, so MediaPipe labels it "Left".  Flip to "Right" if using a mirrored cam.
TARGET_HAND   = "Left"   # tracks your physical right hand on a non-mirrored ZED

# cv2.imshow conflicts with mjpython's Cocoa event loop on macOS.
# Set to True only when running with plain `python` (not `mjpython`).
SHOW_CAMERA  = False

# ── Quaternion helpers ────────────────────────────────────────────────────────
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_ensure_hemi(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Negate q if it is in the opposite hemisphere from ref (avoids filter flips)."""
    return -q if np.dot(q, ref) < 0 else q


# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "robots", "leap_hand", "scene.xml")


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


def _update(data:     mujoco.MjData,
            zed:      ZEDCamera,
            tracker:  StereoHandTracker,
            ik:       IKRetargeter,
            pos_f:    OneEuroFilter,
            joint_f:  OneEuroFilter,
            orient_f: OneEuroFilter,
            mid:      int) -> None:
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
        # ── Stereo path — full pinhole back-projection ────────────────────────
        res_l, res_r = tracker.process(frame_l, frame_r)

        if not (res_l.multi_hand_landmarks and res_r.multi_hand_landmarks):
            _show(frame_l)
            return

        lm_l = res_l.multi_hand_landmarks[0].landmark
        lm_r = res_r.multi_hand_landmarks[0].landmark

        # Epipolar check (Y_OFFSET_PX correction already baked into frames)
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

        # Full 3-D hand position via pinhole back-projection:
        #   1. Median depth over 4 anchor landmarks (wrist + 3 MCPs) → robust Z
        #   2. Back-project wrist pixel with principal-point correction → (X,Y,Z)
        p3d = geo.stereo_hand_3d(lm_l, lm_r, w, h,
                                  depth_min_m=DEPTH_MIN_M,
                                  depth_max_m=DEPTH_MAX_M)
        if p3d is None:
            # All disparities zero/negative — bad stereo frame, skip
            _show(frame_l)
            return

        x_m, y_m, z_m = p3d   # X: right+, Y: down+, Z: forward+

        # Camera frame → sim frame:
        #   sim_x: mirror camera X  (your right = sim +x)
        #   sim_y: camera Z maps to forward depth, offset from neutral
        #   sim_z: invert camera Y  (camera down = sim up)
        sim_x = -x_m
        sim_y =  START_Y + (DEPTH_MID_M - z_m) * DEPTH_SCALE
        sim_z =  START_Z - y_m

        if SHOW_CAMERA:
            cv2.putText(frame_l,
                        f"Z={z_m*100:.0f}cm  X={x_m*100:+.0f}cm  epi={epi_err:.0f}px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    else:
        # ── Monocular path — left frame only ─────────────────────────────────
        res_l, _ = tracker.process(frame_l, frame_r)   # right result ignored

        if not res_l.multi_hand_landmarks:
            if SHOW_CAMERA:
                cv2.putText(frame_l, "No hand", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            _show(frame_l)
            return

        # Reject if the detected hand is not the target hand (avoids mirrored joints).
        label = res_l.multi_handedness[0].classification[0].label  # "Left" or "Right"
        if label != TARGET_HAND:
            if SHOW_CAMERA:
                cv2.putText(frame_l, f"{label} hand ignored", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            _show(frame_l)
            return

        lm_l = res_l.multi_hand_landmarks[0].landmark

        # Principal-point-corrected lateral/vertical.
        # Equivalent to back_project at a fixed assumed depth of START_Y metres.
        # Removes the image-centre bias in the old (lm.x - 0.5) formula.
        cam   = geo.ZED2I
        u_w   = lm_l[0].x * w
        v_w   = lm_l[0].y * h
        sim_x = -(u_w - cam.cx) / cam.fx * START_Y
        sim_y =  START_Y
        sim_z =  START_Z - (v_w - cam.cy) / cam.fy * START_Y

        if SHOW_CAMERA:
            cv2.putText(frame_l, "Monocular IK", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    # ── Mocap position ────────────────────────────────────────────────────────
    raw_pos = np.array([sim_x, sim_y, sim_z])
    data.mocap_pos[mid] = pos_f(raw_pos)
    # ── Wrist orientation ───────────────────────────────────────────────────────────────────
    if WRIST_ORIENT:
        raw_q = palm_quat(lm_l)
        # Ensure quaternion stays in the same hemisphere as the filter state
        # to prevent sudden sign-flip jumps.
        ref   = orient_f._x if orient_f._x is not None else raw_q
        raw_q = _quat_ensure_hemi(raw_q, ref)
        q_ori = orient_f(raw_q)
        q_ori /= np.linalg.norm(q_ori) + 1e-9      # renormalise after filtering
        # Compose: ORIENT_OFFSET rotates the MediaPipe-derived orientation
        # to match the LEAP hand's rest pose in the XML.
        data.mocap_quat[mid] = _quat_mul(ORIENT_OFFSET, q_ori)
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
    ik       = IKRetargeter(model)
    pos_f    = OneEuroFilter(POS_FREQ,    min_cutoff=POS_MC,    beta=POS_BETA)
    joint_f  = OneEuroFilter(JOINT_FREQ,  min_cutoff=JOINT_MC,  beta=JOINT_BETA)
    orient_f = OneEuroFilter(ORIENT_FREQ, min_cutoff=ORIENT_MC, beta=ORIENT_BETA)

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
            _update(data, zed, tracker, ik, pos_f, joint_f, orient_f, mid)

            mujoco.mj_step(model, data)
            v.sync()

            # Allow quitting from the camera window
            if SHOW_CAMERA and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
