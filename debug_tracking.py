"""
debug_tracking.py — Camera-only hand tracking preview. No MuJoCo.

Run with plain python (NOT mjpython):
    python debug_tracking.py

Shows:
  - Left ZED frame with MediaPipe skeleton
  - Right ZED frame with MediaPipe skeleton
  - Depth estimate from stereo disparity
  - Epipolar error

Press Q to quit.
"""

import cv2
import numpy as np

from binocular.camera    import ZEDCamera
from binocular.detectors import StereoHandTracker
import binocular.geometry as geo

CAMERA_ID    = 0
# Tolerance is tight now that the 30 px vertical bias is corrected at the
# image level.  Flip geo.Y_OFFSET_PX's sign in geometry.py if the error
# increases instead of dropping near zero after this change.
EPIPOLAR_TOL = 5

zed     = ZEDCamera(camera_id=CAMERA_ID, y_offset=geo.Y_OFFSET_PX)
tracker = StereoHandTracker()

print("Camera preview running — press Q to quit.")

while True:
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        continue

    res_l, res_r = tracker.process(frame_l, frame_r)

    h, w, _ = frame_l.shape

    # Draw landmarks on both frames
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
        color  = (0, 165, 255)
    elif res_r.multi_hand_landmarks:
        status = "Right only"
        color  = (0, 165, 255)

    cv2.putText(frame_l, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Side-by-side display
    display = np.hstack([frame_l, frame_r])
    # Scale down if too wide for screen
    dh, dw, _ = display.shape
    if dw > 1920:
        scale = 1920 / dw
        display = cv2.resize(display, (int(dw * scale), int(dh * scale)))

    cv2.imshow("ZED Stereo — Hand Tracking Debug (Q to quit)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
cv2.destroyAllWindows()
