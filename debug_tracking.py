"""
debug_tracking.py — Camera-only tracking preview. No MuJoCo.

Run with plain python (NOT mjpython):
    python debug_tracking.py

MODE = "hand"  →  stereo hand tracking  (epipolar + depth debug)
MODE = "arm"   →  pose arm tracking     (shoulder/elbow/wrist world coords)

Press Q to quit.
"""

import cv2
import numpy as np

from vision.camera    import ZEDCamera
from vision.detectors import StereoHandTracker, ArmTracker
import vision.geometry as geo

# ── Toggle ────────────────────────────────────────────────────────────────────
MODE         = "arm"   # "hand" | "arm"
CAMERA_ID    = 0
EPIPOLAR_TOL = 5       # only used in "hand" mode


# ── Setup ─────────────────────────────────────────────────────────────────────
zed = ZEDCamera(camera_id=CAMERA_ID,
                y_offset=geo.Y_OFFSET_PX if MODE == "hand" else 0)

if MODE == "hand":
    tracker = StereoHandTracker()
else:
    tracker = ArmTracker()

print(f"[{MODE.upper()} MODE]  Camera preview running — press Q to quit.")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        continue

    h, w, _ = frame_l.shape

    # ── HAND mode ─────────────────────────────────────────────────────────
    if MODE == "hand":
        res_l, res_r = tracker.process(frame_l, frame_r)

        tracker.draw_landmarks(frame_l, res_l)
        tracker.draw_landmarks(frame_r, res_r)

        status = "No hand"
        color  = (0, 165, 255)

        if res_l.multi_hand_landmarks and res_r.multi_hand_landmarks:
            lm_l = res_l.multi_hand_landmarks[0].landmark
            lm_r = res_r.multi_hand_landmarks[0].landmark

            py_l = int(lm_l[0].y * h)
            py_r = int(lm_r[0].y * h)
            valid, epi_err = geo.check_epipolar_constraint(py_l, py_r,
                                                            tolerance_px=EPIPOLAR_TOL)

            px_l = int(lm_l[0].x * w)
            px_r = int(lm_r[0].x * w)
            z_cm = geo.triangulate_depth(px_l, px_r)

            if valid:
                status = f"OK  epi={epi_err:.0f}px  depth={z_cm:.1f}cm"
                color  = (0, 220, 0)
            else:
                status = f"EPIPOLAR REJECTED  err={epi_err:.0f}px"
                color  = (0, 0, 255)
        elif res_l.multi_hand_landmarks:
            status = "Left only"
        elif res_r.multi_hand_landmarks:
            status = "Right only"

        cv2.putText(frame_l, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        display = np.hstack([frame_l, frame_r])

    # ── ARM mode ──────────────────────────────────────────────────────────
    else:
        result = tracker.process(frame_l)
        tracker.draw_landmarks(frame_l, result)

        if result.pose_world_landmarks:
            lm = result.pose_world_landmarks.landmark

            # Right-arm landmarks
            S = lm[ArmTracker.SHOULDER_R]
            E = lm[ArmTracker.ELBOW_R]
            Wr = lm[ArmTracker.WRIST_R]

            # Overlay world coords and visibility on frame
            lines = [
                (f"shoulder  ({S.x:+.2f}, {S.y:+.2f}, {S.z:+.2f})  vis={S.visibility:.2f}",
                 (0, 220, 0)   if S.visibility > 0.7 else (0, 165, 255)),
                (f"elbow     ({E.x:+.2f}, {E.y:+.2f}, {E.z:+.2f})  vis={E.visibility:.2f}",
                 (0, 220, 0)   if E.visibility > 0.7 else (0, 165, 255)),
                (f"wrist     ({Wr.x:+.2f}, {Wr.y:+.2f}, {Wr.z:+.2f})  vis={Wr.visibility:.2f}",
                 (0, 220, 0)   if Wr.visibility > 0.7 else (0, 165, 255)),
            ]

            # Upper arm and forearm vectors (for angle display)
            ua = np.array([E.x - S.x, E.y - S.y, E.z - S.z])
            fa = np.array([Wr.x - E.x, Wr.y - E.y, Wr.z - E.z])
            n_ua = np.linalg.norm(ua) + 1e-6
            n_fa = np.linalg.norm(fa) + 1e-6
            elbow_angle_deg = float(np.degrees(
                np.arccos(np.clip(np.dot(ua / n_ua, fa / n_fa), -1, 1))
            ))
            lines.append((f"elbow angle = {elbow_angle_deg:.0f} deg", (200, 200, 200)))

            for i, (text, color) in enumerate(lines):
                cv2.putText(frame_l, text, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame_l, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        display = frame_l

    # ── Display ───────────────────────────────────────────────────────────
    dh, dw = display.shape[:2]
    if dw > 1920:
        scale   = 1920 / dw
        display = cv2.resize(display, (int(dw * scale), int(dh * scale)))

    cv2.imshow(f"ZED — {MODE.upper()} tracking debug  (Q to quit)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
if MODE == "arm":
    tracker.close()
zed.close()
cv2.destroyAllWindows()
