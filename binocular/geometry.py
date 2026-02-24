# geometry.py
#
# ZED 2i stereo constants — tune these if depth feels wrong.
#
# BASELINE_CM:     physical distance between the two lenses (~12 cm for ZED 2i).
# FOCAL_LENGTH_PX: camera focal length in pixels at your capture resolution.
#   ZED 2i @ 720p SBS (each half = 1280×720): factory calibration ≈ 700 px.
#   If depth reads consistently too far,  increase this value.
#   If depth reads consistently too close, decrease this value.
#
# Y_OFFSET_PX: vertical pixel offset of the right lens relative to the left.
#   A constant ~30 px epipolar error means the two sensors are not level.
#   ZEDCamera.get_frames() shifts the right frame by this many pixels so that
#   all downstream geometry sees perfectly aligned rows.
#   Positive → right frame is shifted DOWN  (right lens sits higher).
#   Negative → right frame is shifted UP    (right lens sits lower).
#   Flip the sign if the error increases instead of disappearing after you apply it.

BASELINE_CM      = 12.0
FOCAL_LENGTH_PX  = 700.0
Y_OFFSET_PX      = 30      # pixels — dial in the sign/magnitude to your camera

def check_epipolar_constraint(y_left, y_right, tolerance_px=5):
    """Return (is_valid, error_px).  Invalid when MediaPipe hallucinates a hand.

    The Y_OFFSET_PX vertical correction is already baked into the frames that
    ZEDCamera.get_frames() returns, so this function just checks the residual.
    Tolerance has been tightened from 15 px → 5 px now that the systematic
    30 px bias is removed at the image level.
    """
    error = abs(y_left - y_right)
    return error <= tolerance_px, error


def triangulate_depth(x_left, x_right):
    """Return depth in cm via Z = (f * B) / disparity, or None if disparity <= 0."""
    disparity = x_left - x_right
    return None if disparity <= 0 else (FOCAL_LENGTH_PX * BASELINE_CM) / disparity