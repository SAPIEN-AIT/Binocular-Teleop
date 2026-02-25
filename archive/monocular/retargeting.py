import numpy as np

# MediaPipe landmark indices (21 total)
#         4
#         |
#    3----2
#         |
#    1----0 (wrist)

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

def angle_between(a, b, c):
    """
    Returns the angle at point b, formed by vectors b->a and b->c.
    All points are (x, y) tuples.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)

def landmark_to_point(lm):
    return (lm.x, lm.y)

def compute_finger_angles(lm, mcp, pip, dip, tip):
    mcp_angle = angle_between(
        landmark_to_point(lm[WRIST]),
        landmark_to_point(lm[mcp]),
        landmark_to_point(lm[pip])
    )
    pip_angle = angle_between(
        landmark_to_point(lm[mcp]),
        landmark_to_point(lm[pip]),
        landmark_to_point(lm[dip])
    )
    dip_angle = angle_between(
        landmark_to_point(lm[pip]),
        landmark_to_point(lm[dip]),
        landmark_to_point(lm[tip])
    )

    # Invert: angle_between returns pi when straight, 0 when closed.
    # LEAP expects 0 = open, positive = closed.
    mcp_angle = np.clip(np.pi - mcp_angle, 0, 1.57)
    pip_angle = np.clip(np.pi - pip_angle, 0, 1.57)
    dip_angle = np.clip(np.pi - dip_angle, 0, 1.57)

    return mcp_angle, pip_angle, dip_angle

def wrist_to_world_pos(landmarks, z_fixed=0.12):
    """
    Maps the wrist 2D image position to a sim world position.
    x, y come from MediaPipe (normalized 0→1 in image space).
    z stays fixed — monocular has no depth, binocular will fix this.

    Coordinate mapping:
      image x (0=left,  1=right) → sim x  (mirrored for natural feel)
      image y (0=top,   1=bot)   → sim y  (inverted: up in image = forward in sim)
      depth                      → sim z  (fixed height above floor)
    """
    wrist = landmarks[WRIST]

    sim_x = -(wrist.x - 0.5) * 0.4   # mirror so moving right moves hand right
    sim_y =  (0.5 - wrist.y) * 0.3   # invert: image y grows down, sim y grows up
    sim_z =  z_fixed

    return [sim_x, sim_y, sim_z]


def landmarks_to_joints(landmarks):
    """
    Takes 21 MediaPipe landmarks, returns 16 LEAP joint angles.

    LEAP actuator order per finger: [mcp, rot, pip, dip]
    We control mcp, pip, dip and lock rot=0 (no splay).
    Fingers: index, middle, ring, thumb (no pinky on LEAP).
    """
    lm = landmarks

    index  = compute_finger_angles(lm, INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP)
    middle = compute_finger_angles(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    ring   = compute_finger_angles(lm, RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP)
    thumb  = compute_finger_angles(lm, THUMB_CMC,  THUMB_MCP,  THUMB_IP,   THUMB_TIP)

    # Order: mcp, rot=0, pip, dip — for each of the 4 fingers
    joints = [
        index[0],  0.0, index[1],  index[2],
        middle[0], 0.0, middle[1], middle[2],
        ring[0],   0.0, ring[1],   ring[2],
        thumb[0],  0.0, thumb[1],  thumb[2],
    ]

    return joints