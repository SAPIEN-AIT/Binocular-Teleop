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
import multiprocessing as _mp
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
CAMERA_ID    = 0       # 0 = webcam / seule caméra détectée. Mettre 1 quand la ZED est branchée.
N_SUBSTEPS   = 8       # physics sub-steps per vision frame (8 × 2ms = 16ms, ~80% tracking)

# ── Mode toggle ───────────────────────────────────────────────────────────────
# False = monocular (left frame only, Y fixed) → tune IK first.
# True  = stereo    (epipolar + triangulated depth) → requires ZED camera.
STEREO_DEPTH = True

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
DEPTH_SCALE  = 3.0     # m of sim-Y movement per m of depth change (5× previous)
START_Y      = 0.30    # initial sim Y (forward) of the hand proxy — also used as fixed Y
START_Z      = 0.30    # initial sim Z (height) of the hand proxy

# Epipolar constraint — only checked when STEREO_DEPTH = True
EPIPOLAR_TOL = 40      # px (relaxed — tighten once Y_OFFSET_PX is tuned for your unit)

# ── Wrist Z-rotation (via mocap_quat) ────────────────────────────────────────
WRIST_SCALE    = 0.5    # gain on detected angle delta
WRIST_DEADZONE = 0.10   # rad (~6°) — noise below this ignored
WRIST_MAX_RAD  = 1.2    # ±69° max clamp

# One Euro Filter for wrist angle (1-dim)
WRIST_FREQ     = 30.0
WRIST_MC       = 0.3
WRIST_BETA     = 0.01

# Rest orientation of the hand (unchanged from last push)
BASE_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # Rx(180°): palm facing up

# ── Handedness filter ─────────────────────────────────────────────────────────────────────
# ZED is a non-mirrored camera: your RIGHT hand appears on the LEFT side of the
# image, so MediaPipe labels it "Left".  Flip to "Right" if using a mirrored cam.
TARGET_HAND   = "Left"   # tracks your physical right hand on a non-mirrored ZED

# cv2.imshow conflicts with mjpython's Cocoa event loop on macOS.
# Set to True only when running with plain `python` (not `mjpython`).
SHOW_CAMERA  = True

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

    data.mocap_pos[mid]  = pos
    data.mocap_quat[mid] = BASE_QUAT.copy()

    jid  = model.joint("palm_free").id
    addr = model.jnt_qposadr[jid]
    data.qpos[addr:addr+3] = pos
    data.qpos[addr+3:addr+7] = BASE_QUAT

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

    Left camera drives finger retargeting (always).
    When STEREO_DEPTH=True, both cameras triangulate wrist depth for sim-Y.
    If stereo fails on a frame, the last good position is kept but fingers
    still update — no frame is ever fully dropped.
    """
    global _wrist_ref_angle, _wrist_calib_count
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        return

    h, w, _ = frame_l.shape
    res_l, res_r = tracker.process(frame_l, frame_r)

    # ── Left camera must see a hand to do anything ────────────────────────
    if not res_l.multi_hand_landmarks:
        _wrist_ref_angle = None
        _wrist_calib_count = 0
        orient_f.reset()
        data.mocap_quat[mid] = BASE_QUAT.copy()
        if SHOW_CAMERA:
            cv2.putText(frame_l, "L: NO HAND", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if frame_r is not None:
                r_det = "R: HAND" if res_r.multi_hand_landmarks else "R: NO HAND"
                r_col = (0, 220, 0) if res_r.multi_hand_landmarks else (0, 0, 255)
                cv2.putText(frame_r, r_det, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, r_col, 2)
                _show(np.hstack([frame_l, frame_r]))
            else:
                _show(frame_l)
        else:
            _show(frame_l)
        return

    lm_l = res_l.multi_hand_landmarks[0].landmark
    cam   = geo.ZED2I

    # ── Wrist position: lateral + vertical from left camera ───────────────
    u_w   = lm_l[0].x * w
    v_w   = lm_l[0].y * h
    sim_x = -(u_w - cam.cx) / cam.fx * START_Y
    sim_z =  START_Z - (v_w - cam.cy) / cam.fy * START_Y

    # ── Depth axis (sim Y): stereo triangulation or fixed ─────────────────
    sim_y      = START_Y
    depth_cm   = None          # None = no stereo depth available
    hud_mode   = "MONO [L]"   # displayed mode label
    hud_col    = (0, 165, 255) # orange = mono
    hud_detail = ""

    if STEREO_DEPTH and res_r.multi_hand_landmarks:
        lm_r = res_r.multi_hand_landmarks[0].landmark
        py_l = int(lm_l[0].y * h)
        py_r = int(lm_r[0].y * h)
        valid, epi_err = geo.check_epipolar_constraint(
            py_l, py_r, tolerance_px=EPIPOLAR_TOL)

        if valid:
            p3d = geo.stereo_hand_3d(lm_l, lm_r, w, h,
                                      depth_min_m=DEPTH_MIN_M,
                                      depth_max_m=DEPTH_MAX_M)
            if p3d is not None:
                x_m, y_m, z_m = p3d
                sim_x = -x_m
                sim_y = START_Y + (DEPTH_MID_M - z_m) * DEPTH_SCALE
                sim_z = START_Z - y_m
                depth_cm = z_m * 100
                hud_mode   = "STEREO [L+R]"
                hud_col    = (0, 220, 0)
                hud_detail = f"epi={epi_err:.0f}px"
            else:
                hud_detail = f"bad disparity epi={epi_err:.0f}px"
        else:
            hud_detail = f"epi REJECTED err={epi_err:.0f}px"

    # ── Mocap position ────────────────────────────────────────────────────
    raw_pos = np.array([sim_x, sim_y, sim_z])
    data.mocap_pos[mid] = pos_f(raw_pos)

    # ── Wrist Z-rotation (pure Rz on mocap_quat) ────────────────────────
    idx_mcp  = lm_l[5]
    ring_mcp = lm_l[13]
    raw_angle = np.arctan2(ring_mcp.y - idx_mcp.y, ring_mcp.x - idx_mcp.x)

    if _wrist_ref_angle is None:
        _wrist_ref_angle = raw_angle
        _wrist_calib_count = 1
    elif _wrist_calib_count < 15:
        err = (raw_angle - _wrist_ref_angle + np.pi) % (2 * np.pi) - np.pi
        _wrist_ref_angle += 0.2 * err
        _wrist_calib_count += 1

    delta = (raw_angle - _wrist_ref_angle + np.pi) % (2 * np.pi) - np.pi
    delta = float(np.clip(delta, -0.9, 0.9))
    delta = float(orient_f(np.array([delta]))[0])
    if abs(delta) < WRIST_DEADZONE:
        delta = 0.0
    wrist_z = float(np.clip(delta * WRIST_SCALE, -WRIST_MAX_RAD, WRIST_MAX_RAD))

    half = wrist_z / 2.0
    q_z = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    data.mocap_quat[mid] = _quat_mul(q_z, BASE_QUAT)

    # ── Direct angle retargeting (always from left camera) ────────────────
    q_raw    = ik.retarget(None, lm_l)
    q_smooth = joint_f(q_raw)
    data.ctrl[:] = q_smooth

    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, res_l)

        # Top-left: mode indicator
        cv2.putText(frame_l, hud_mode, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2)
        if hud_detail:
            cv2.putText(frame_l, hud_detail, (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        # Top-right: depth readout (large)
        if depth_cm is not None:
            depth_str = f"DEPTH: {depth_cm:.0f} cm"
            d_col = (0, 220, 0)
        else:
            depth_str = "DEPTH: ---"
            d_col = (0, 165, 255)
        txt_size = cv2.getTextSize(depth_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame_l, depth_str, (w - txt_size[0] - 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_col, 2)

        # Show both cameras side by side with detection status
        if frame_r is not None:
            tracker.draw_landmarks(frame_r, res_r)
            r_det = "R: HAND" if res_r.multi_hand_landmarks else "R: NO HAND"
            r_col = (0, 220, 0) if res_r.multi_hand_landmarks else (0, 0, 255)
            cv2.putText(frame_r, r_det, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_col, 2)
            display = np.hstack([frame_l, frame_r])
        else:
            display = frame_l

    _show(display if SHOW_CAMERA else frame_l)


_wrist_ref_angle = None
_wrist_calib_count = 0
_frame_q = None
_show_counter = 0
_SHOW_EVERY = 3        # send 1 frame out of 3 to the viewer (saves CPU + queue bandwidth)
_VIEWER_SCALE = 0.5    # downscale factor for the viewer frame


def _show(frame):
    """Send a downscaled frame to the viewer subprocess every N calls."""
    global _show_counter
    if not SHOW_CAMERA or _frame_q is None:
        return
    _show_counter += 1
    if _show_counter % _SHOW_EVERY != 0:
        return
    small = cv2.resize(frame, None, fx=_VIEWER_SCALE, fy=_VIEWER_SCALE,
                       interpolation=cv2.INTER_NEAREST)
    if _frame_q.full():
        try:
            _frame_q.get_nowait()
        except Exception:
            pass
    try:
        _frame_q.put_nowait(small)
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global _frame_q

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
    orient_f = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)

    # Spawn hand at rest position
    _init_hand(model, data)

    # Camera viewer in a separate lightweight process (only imports cv2,
    # NOT mujoco — avoids the Cocoa / OpenGL conflict with mjpython on macOS).
    viewer_proc = None
    if SHOW_CAMERA:
        from _camera_viewer import viewer_loop
        ctx = _mp.get_context("spawn")
        _frame_q = ctx.Queue(maxsize=2)
        viewer_proc = ctx.Process(target=viewer_loop, args=(_frame_q,), daemon=True)
        viewer_proc.start()

    print("─" * 60)
    print("  Binocular Hand Teleoperation (Direct Angle Retargeting)")
    print("  Move your right hand in front of the ZED camera.")
    print("  Press  Q  in the camera window  or  ESC  in the")
    print("  MuJoCo viewer to quit.")
    print("─" * 60)

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            _update(data, zed, tracker, ik, pos_f, joint_f, orient_f, mid)

            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
            v.sync()

    # Clean shutdown
    if viewer_proc is not None and _frame_q is not None:
        _frame_q.put(None)
        viewer_proc.join(timeout=3)

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
