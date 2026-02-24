"""
Direct angle-based retargeting: MediaPipe 3D landmarks → LEAP joint angles.

Why not Jacobian IK targets?
  Phase 0 has no wrist orientation tracking, so we can't correctly map a
  MediaPipe image-plane vector to a 3-D world-space fingertip target — the
  required "natural extension" direction of each finger in world space changes
  every time the hand moves or rotates.

  Direct angle mapping avoids the problem entirely:
    1. Compute the bend angle at each MediaPipe joint from its two adjacent
       bone vectors (using all three x/y/z coordinates, so depth is captured).
    2. Scale the angle into the corresponding LEAP joint range.
    3. Done — no plan_data, no IK solver, no world-frame math.

Interface is preserved so mujoco_teleop.py needs no changes.

MediaPipe landmark indices (21 total)
--------------------------------------
WRIST=0
THUMB : CMC=1  MCP=2  IP=3  TIP=4
INDEX : MCP=5  PIP=6  DIP=7  TIP=8
MIDDLE: MCP=9  PIP=10 DIP=11 TIP=12
RING  : MCP=13 PIP=14 DIP=15 TIP=16
"""

import numpy as np
import mujoco

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
WRIST       = 0
THUMB_CMC   = 1; THUMB_MCP_LM = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP   = 5; INDEX_PIP    = 6; INDEX_DIP = 7; INDEX_TIP  = 8
MIDDLE_MCP  = 9; MIDDLE_PIP   = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP    = 14; RING_DIP   = 15; RING_TIP   = 16

# ── Tuning ─────────────────────────────────────────────────────────────────────
# Maximum observed bend angle in MediaPipe 3-D (≈ 130-135°, clamp generously)
_MP_MAX = 2.3   # radians

# Practical LEAP upper limits (slightly below XML max to stay away from limits)
_MCP_MAX  = 2.0   # LEAP xml max 2.23
_PIP_MAX  = 1.7   # LEAP xml max 1.885
_DIP_MAX  = 1.8   # LEAP xml max 2.042
_ROT_LIM  = 0.6   # abduction clamp (xml ±1.047)
_CMC_MAX  = 1.8   # th_cmc
_AXL_MAX  = 1.8   # th_axl
_TMCP_MAX = 2.0   # th_mcp
_IPL_MAX  = 1.5   # th_ipl


# ── Helpers ────────────────────────────────────────────────────────────────────
def _lm3(lm, i: int) -> np.ndarray:
    """Landmark → (x, y, z) array. Uses MediaPipe's pseudo-3D z (depth)."""
    return np.array([lm[i].x, lm[i].y, lm[i].z])


def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between two bone vectors at a joint.

    0   → straight finger (v1 and v2 point the same direction)
    π/2 → 90° bend
    π   → fully curled (theoretical)

    All three MediaPipe coordinates (x, y, z) are used, so finger depth
    contributes to the angle correctly.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _scale(angle: float, leap_max: float) -> float:
    """Map MediaPipe bend ∈ [0, _MP_MAX] → LEAP angle ∈ [0, leap_max]."""
    return float(np.clip(angle / _MP_MAX * leap_max, 0.0, leap_max))


# ── Main class ─────────────────────────────────────────────────────────────────
class IKRetargeter:
    """
    Maps MediaPipe hand landmarks → 16 LEAP joint angles.

    plan_data and palm_xpos are accepted for API compatibility but are unused;
    retargeting is purely angle-based and requires no world-frame information.

    Usage
    -----
        ik = IKRetargeter(model)
        q  = ik.retarget(plan_data, landmarks, palm_xpos)  # 16 joint angles
    """

    def __init__(self, model: mujoco.MjModel,
                 n_iters: int = 15,   # kept for API compat (unused)
                 step: float  = 0.5): # kept for API compat (unused)
        self.model = model

    # ------------------------------------------------------------------
    def retarget(self,
                 plan_data,           # unused — kept for API compat
                 lm,
                 palm_xpos=None       # unused — kept for API compat
                 ) -> np.ndarray:
        """
        Compute 16 LEAP joint angles from a MediaPipe landmark list.

        Args:
            plan_data:  ignored (API compat)
            lm:         result.multi_hand_landmarks[0].landmark  (21 entries)
            palm_xpos:  ignored (API compat)

        Returns:
            q (16,): desired joint angles in LEAP actuator order
                     [if_mcp, if_rot, if_pip, if_dip,
                      mf_mcp, mf_rot, mf_pip, mf_dip,
                      rf_mcp, rf_rot, rf_pip, rf_dip,
                      th_cmc, th_axl, th_mcp, th_ipl]
        """
        q = np.zeros(16)

        w  = _lm3(lm, WRIST)
        mm = _lm3(lm, MIDDLE_MCP)   # used as lateral reference for rot joints

        # ── Index finger (0-3: mcp, rot, pip, dip) ────────────────────────────
        im  = _lm3(lm, INDEX_MCP);  ip  = _lm3(lm, INDEX_PIP)
        idd = _lm3(lm, INDEX_DIP);  it  = _lm3(lm, INDEX_TIP)

        q[0] = _scale(_bend(im - w,   ip  - im), _MCP_MAX)   # if_mcp
        # rot: index spreads laterally away from middle (left in image = positive rot)
        spread_i = float(np.clip((im[0] - mm[0]) * 8.0, -_ROT_LIM, _ROT_LIM))
        q[1] = spread_i                                        # if_rot
        q[2] = _scale(_bend(ip - im,  idd - ip),  _PIP_MAX)   # if_pip
        q[3] = _scale(_bend(idd - ip, it  - idd), _DIP_MAX)   # if_dip

        # ── Middle finger (4-7) ────────────────────────────────────────────────
        mmp = _lm3(lm, MIDDLE_MCP); mpi = _lm3(lm, MIDDLE_PIP)
        mdi = _lm3(lm, MIDDLE_DIP); mti = _lm3(lm, MIDDLE_TIP)

        q[4] = _scale(_bend(mmp - w,   mpi - mmp), _MCP_MAX)  # mf_mcp
        q[5] = 0.0                                             # mf_rot (centred)
        q[6] = _scale(_bend(mpi - mmp, mdi - mpi), _PIP_MAX)  # mf_pip
        q[7] = _scale(_bend(mdi - mpi, mti - mdi), _DIP_MAX)  # mf_dip

        # ── Ring finger (8-11) ─────────────────────────────────────────────────
        rm  = _lm3(lm, RING_MCP);  rp  = _lm3(lm, RING_PIP)
        rd  = _lm3(lm, RING_DIP);  rt  = _lm3(lm, RING_TIP)

        q[8]  = _scale(_bend(rm - w,  rp - rm), _MCP_MAX)     # rf_mcp
        spread_r = float(np.clip((mm[0] - rm[0]) * 8.0, -_ROT_LIM, _ROT_LIM))
        q[9]  = spread_r                                       # rf_rot
        q[10] = _scale(_bend(rp - rm, rd - rp), _PIP_MAX)     # rf_pip
        q[11] = _scale(_bend(rd - rp, rt - rd), _DIP_MAX)     # rf_dip

        # ── Thumb (12-15: th_cmc, th_axl, th_mcp, th_ipl) ────────────────────
        tc  = _lm3(lm, THUMB_CMC)
        tm  = _lm3(lm, THUMB_MCP_LM)
        tip = _lm3(lm, THUMB_IP)
        tt  = _lm3(lm, THUMB_TIP)

        # th_cmc: metacarpal curl — how far thumb rises from palm
        q[12] = _scale(_bend(tc - w,  tm - tc),  _CMC_MAX)    # th_cmc

        # th_axl: opposition — how much thumb tip crosses toward index
        # When thumb tip is to the LEFT of the index MCP (x smaller), it opposes
        opposition = float(np.clip((im[0] - tt[0]) * 6.0, 0.0, 1.0))
        q[13] = opposition * _AXL_MAX                          # th_axl

        q[14] = _scale(_bend(tm - tc,  tip - tm), _TMCP_MAX)  # th_mcp
        q[15] = _scale(_bend(tip - tm, tt - tip), _IPL_MAX)   # th_ipl

        return q
