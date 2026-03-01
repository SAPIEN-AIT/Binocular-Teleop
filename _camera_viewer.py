"""
Lightweight camera viewer process — imports ONLY cv2.

Spawned by teleop_leap.py so that cv2.imshow runs in its own Cocoa
event loop, avoiding the conflict with mjpython on macOS.
"""


def viewer_loop(q, reset_flag=None):
    import cv2
    while True:
        item = q.get()
        if item is None:
            break
        cv2.imshow("ZED Left — Binocular Teleop", item)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and reset_flag is not None:
            reset_flag.value = 1
    cv2.destroyAllWindows()
