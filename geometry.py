# geometry.py

# ZED 2i Approximate Constants (We will calibrate these later)
BASELINE_CM = 12.0          # Distance between the two lenses
FOCAL_LENGTH_PX = 700.0     # Estimated focal length for 720p

def check_epipolar_constraint(y_left, y_right, tolerance_px=15):
    """
    Checks if the vertical pixels align. 
    If they don't, MediaPipe made a mistake (hallucination).
    """
    error = abs(y_left - y_right)
    is_valid = error <= tolerance_px
    return is_valid, error

def triangulate_depth(x_left, x_right):
    """
    Calculates the Z depth in centimeters.
    """
    disparity = x_left - x_right
    
    # Prevent division by zero or impossible negative disparity
    if disparity <= 0:
        return None
        
    # The core stereo formula: Z = (f * B) / d
    depth_z = (FOCAL_LENGTH_PX * BASELINE_CM) / disparity
    return depth_z