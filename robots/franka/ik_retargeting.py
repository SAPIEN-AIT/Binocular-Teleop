"""
robots/franka/ik_retargeting.py
Jacobian IK retargeting: MediaPipe Pose world landmarks → 7 Franka FR3 joint angles.

Strategy (new — replaces the old direct-angle approach)
--------------------------------------------------------
Instead of mapping joint angles one-to-one, we:

  1. Extract the human wrist 3-D position in MediaPipe world space
     (metric, hip-centred; available via pose_world_landmarks).

  2. Map it to a desired end-effector (EE) position in the robot's
     MuJoCo world frame, using a simple linear workspace transform
     anchored to the home pose.

  3. Solve IK via Damped Least Squares (DLS):

       δq  = Jᵀ(JJᵀ + λ²I)⁻¹ δx          ← position-only DLS step
            + (I – J⁺J) · k · (q_home – q) ← null-space → natural elbow posture

     This uses MuJoCo's mj_jac() to get the 3×nv position Jacobian at the
     current EE site, then extracts the 7 columns corresponding to the Franka
     joints. A separate MjData is used so FK calls never disturb the live sim.

  4. Run N_ITERS iterations per frame (warm-started from the previous solution)
     so the EE converges tightly to the target.

Key advantage over direct-angle retargeting
-------------------------------------------
• All 7 DOF are active — the previously locked "twist" joints now contribute
  naturally to resolving redundancy.
• Motion lives in Cartesian space: your wrist moves → robot EE moves.
• Null-space attraction keeps the elbow in a natural upright configuration.

Coordinate conventions
-----------------------
MediaPipe world landmarks (pose_world_landmarks):
    x = right (person's right),  y = down (image convention),  z = toward camera
    origin = centre of hips,     units = metres

Robot MuJoCo world (used consistently with the old _to_mujoco helper):
    x = right (= mp x),  y = forward (= mp z),  z = up (= −mp y)

Neutral arm position
--------------------
When the user holds their arm in the "ready" position — slightly forward of
their shoulder, at shoulder height — the robot should rest at its home pose.

Default neutral: wrist−shoulder ≈ (0, 0, 0.35) in mp frame
    → (0, 0.35, 0) in robot frame (35 cm forward, same height as shoulder)

Tune NEUTRAL_MP in FrankaJacobianIK if your setup is different.
"""

import numpy as np
import mujoco

# ── MediaPipe Pose landmark indices ───────────────────────────────────────────
SHOULDER_R = 12
ELBOW_R    = 14
WRIST_R    = 16
INDEX_R    = 20   # index fingertip — used for wrist-flex fallback

# ── Franka FR3 joint ranges (radians, from fr3v2.xml) ─────────────────────────
_J_RANGE = np.array([
    [-2.7437,  2.7437],   # joint1: shoulder pan
    [-1.7837,  1.7837],   # joint2: shoulder lift
    [-2.9007,  2.9007],   # joint3: upper-arm twist
    [-3.0421, -0.1518],   # joint4: elbow flex (always negative = bent)
    [-2.8065,  2.8065],   # joint5: forearm twist
    [ 0.5445,  4.5169],   # joint6: wrist flex
    [-3.0159,  3.0159],   # joint7: wrist rotation
])

# ── Home configuration (keyframe "home" in fr3v2.xml) ────────────────────────
HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])


# ── Frame conversion ──────────────────────────────────────────────────────────

def _mp_to_robot(v_mp: np.ndarray) -> np.ndarray:
    """
    Convert a vector from MediaPipe world frame to the robot world frame.

    MediaPipe world: x = right, y = down (image convention), z = toward camera
    Robot (MuJoCo):  x = right, y = forward (= mp z),        z = up (= −mp y)

    Consistent with the legacy _to_mujoco() helper used in the old retargeter.
    """
    return np.array([v_mp[0], v_mp[2], -v_mp[1]])


def _lm3(lm, i: int) -> np.ndarray:
    """Extract landmark i as a (3,) NumPy array."""
    return np.array([lm[i].x, lm[i].y, lm[i].z])


# ── Main IK class ─────────────────────────────────────────────────────────────

class FrankaJacobianIK:
    """
    Damped Least Squares Jacobian IK for Franka FR3 teleoperation.

    Maps the human wrist 3-D world position to 7 Franka joint angles via:

        δq = Jᵀ(JJᵀ + λ²I)⁻¹ δx          ← DLS position step
           + (I – J⁺J) · k · (q_home – q)  ← null-space → natural posture

    N_ITERS iterations are run per call (warm-started) so the EE converges
    tightly to the target within one vision frame.

    Usage
    -----
        ik = FrankaJacobianIK(model)
        q  = ik.solve(wrist_world, shoulder_world)   # (7,) joint angles
    """

    # ── Tunable parameters ────────────────────────────────────────────────────
    WORKSPACE_SCALE = 1.0    # 1.0 → 1 m of human motion maps to 1 m robot motion
    DAMPING         = 0.05   # DLS λ: higher → smoother near singularities
    NULL_GAIN       = 0.1    # null-space gain: strength of home-posture attraction
    N_ITERS         = 10     # IK iterations per solve() call
    CONV_TOL        = 5e-4   # early-exit convergence threshold [m]

    # Neutral arm vector in MediaPipe world (wrist − shoulder) for the
    # "teleoperation ready" posture that maps to the robot's home pose.
    #   x = 0   → arm not swept left/right from shoulder
    #   y = 0   → arm at shoulder height (y=0 = no height difference)
    #   z = 0.35 → arm 35 cm toward camera (slightly forward)
    # Adjust if you prefer a different starting posture.
    NEUTRAL_MP = np.array([0.0, 0.0, 0.35])

    DEBUG = False

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model    = model
        # Separate MjData for IK FK calls — never touches the live sim state
        self._ik_data = mujoco.MjData(model)

        # End-effector site (added to fr3v2_link8 in fr3v2.xml)
        self._ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        if self._ee_id < 0:
            raise RuntimeError(
                "Site 'end_effector' not found in model. "
                "Add <site name='end_effector'/> inside fr3v2_link8 in fr3v2.xml."
            )

        # DOF and qpos addresses for the 7 Franka joints (safe for any nv)
        self._dof_ids  = np.array([
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3v2_joint{i}")
            ]
            for i in range(1, 8)
        ], dtype=np.int32)

        self._qpos_ids = np.array([
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3v2_joint{i}")
            ]
            for i in range(1, 8)
        ], dtype=np.int32)

        # Compute home EE position via FK
        self._ik_data.qpos[self._qpos_ids] = HOME
        mujoco.mj_forward(model, self._ik_data)
        self.home_ee_pos = self._ik_data.site_xpos[self._ee_id].copy()

        # Neutral offset in robot frame: subtract so neutral arm → home EE
        self._neutral_robot = _mp_to_robot(self.NEUTRAL_MP)

        # Current joint config — maintained across frames for warm-starting IK
        self._q = HOME.copy()

        if self.DEBUG:
            print(f"[FrankaJacobianIK] home EE pos    = {self.home_ee_pos}")
            print(f"[FrankaJacobianIK] neutral (robot) = {self._neutral_robot}")
            print(f"[FrankaJacobianIK] DOF ids         = {self._dof_ids}")

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(self,
              wrist_world:    np.ndarray,
              shoulder_world: np.ndarray) -> np.ndarray:
        """
        Solve IK for the given human wrist 3-D position.

        Parameters
        ----------
        wrist_world    : (3,) wrist position in MediaPipe world frame
        shoulder_world : (3,) shoulder position in MediaPipe world frame
            Both in metres, hip-centred, mp convention (x=right, y=down, z=toward_cam)

        Returns
        -------
        q : (7,) Franka joint angles (actuator order joint1…joint7)
        """
        target  = self._target_pos(wrist_world, shoulder_world)
        self._q = self._ik_loop(target)
        return self._q.copy()

    def last_q(self) -> np.ndarray:
        """Last solved joint config — safe fallback when tracking is lost."""
        return self._q.copy()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _target_pos(self,
                    wrist:    np.ndarray,
                    shoulder: np.ndarray) -> np.ndarray:
        """
        Map human wrist position → desired EE position in robot world frame.

        The arm vector (wrist − shoulder) is converted to robot frame and
        scaled by WORKSPACE_SCALE.  When the arm is at NEUTRAL_MP, the target
        equals home_ee_pos so the robot rests at its home configuration.
        """
        delta_mp    = wrist - shoulder
        delta_robot = _mp_to_robot(delta_mp) * self.WORKSPACE_SCALE
        return self.home_ee_pos + delta_robot - self._neutral_robot

    def _ik_loop(self, target: np.ndarray) -> np.ndarray:
        """
        Run up to N_ITERS DLS IK steps, warm-started from self._q.

        Each iteration:
          1. FK → current EE position
          2. Compute position Jacobian J (3×7) via mj_jac
          3. DLS step:  δq = Jᵀ(JJᵀ + λ²I)⁻¹ (target − ee)
          4. Null-space: (I − J⁺J) · k · (HOME − q)
          5. Clamp to joint limits
        """
        q    = self._q.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(self.N_ITERS):
            # 1. Forward kinematics on the IK data copy
            self._ik_data.qpos[self._qpos_ids] = q
            mujoco.mj_forward(self.model, self._ik_data)
            ee_pos = self._ik_data.site_xpos[self._ee_id].copy()

            # 2. Position error
            err = target - ee_pos
            if np.linalg.norm(err) < self.CONV_TOL:
                break

            # 3. Position Jacobian — extract 3×7 columns for Franka joints
            mujoco.mj_jac(self.model, self._ik_data,
                          jacp, jacr,
                          ee_pos,
                          self.model.site_bodyid[self._ee_id])
            J = jacp[:, self._dof_ids]          # 3 × 7

            # 4. DLS: δq = Jᵀ (JJᵀ + λ²I)⁻¹ err
            lam2 = self.DAMPING ** 2
            A    = J @ J.T + lam2 * np.eye(3)   # 3×3 — cheap invert
            dq   = J.T @ np.linalg.solve(A, err)

            # 5. Null-space: stay near home posture
            J_pinv    = J.T @ np.linalg.inv(A)
            N         = np.eye(7) - J_pinv @ J
            null_term = N @ (self.NULL_GAIN * (HOME - q))

            # 6. Step and clamp
            q = q + dq + null_term
            for i in range(7):
                q[i] = float(np.clip(q[i], _J_RANGE[i, 0], _J_RANGE[i, 1]))

        if self.DEBUG:
            self._ik_data.qpos[self._qpos_ids] = q
            mujoco.mj_forward(self.model, self._ik_data)
            residual = np.linalg.norm(target - self._ik_data.site_xpos[self._ee_id])
            names = ['pan', 'lift', 'twist', 'elbow', 'f-twist', 'w-flex', 'w-rot']
            parts = " ".join(f"{n}={v:+.2f}" for n, v in zip(names, q))
            print(f"IK err={residual*1000:.1f}mm  {parts}")

        return q


# ── Legacy direct-angle retargeter (kept for reference) ──────────────────────
# The class below is the old direct-angle approach.  It is no longer imported
# by teleop_franka.py but kept here so git history stays meaningful.

class _LegacyFrankaIKRetargeter:
    """
    [LEGACY] Direct geometric angle retargeting.
    Superseded by FrankaJacobianIK.  Not used in the main teleoperation loop.
    """

    _SHOULDER_PAN_MAX  = np.radians(80)
    _SHOULDER_LIFT_MAX = np.radians(80)
    _ELBOW_BEND_MAX    = np.radians(150)
    _WRIST_BEND_MAX    = np.radians(90)

    def __init__(self, model: mujoco.MjModel):
        self.model = model

    @staticmethod
    def _to_mujoco(v_cam: np.ndarray) -> np.ndarray:
        return np.array([v_cam[0], v_cam[2], -v_cam[1]])

    @staticmethod
    def _bend(v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

    def retarget(self, pose_result) -> np.ndarray:
        if pose_result is None or pose_result.pose_world_landmarks is None:
            return HOME.copy()
        lm = pose_result.pose_world_landmarks.landmark
        if lm[SHOULDER_R].visibility < 0.5 or lm[ELBOW_R].visibility < 0.5 or \
           lm[WRIST_R].visibility < 0.5:
            return HOME.copy()
        shoulder = _lm3(lm, SHOULDER_R)
        elbow    = _lm3(lm, ELBOW_R)
        wrist    = _lm3(lm, WRIST_R)
        hand_dir_mp = _lm3(lm, INDEX_R) - wrist if lm[INDEX_R].visibility > 0.5 \
                      else wrist - elbow
        upper_arm = self._to_mujoco(elbow - shoulder)
        forearm   = self._to_mujoco(wrist - elbow)
        hand_dir  = self._to_mujoco(hand_dir_mp)
        ua_horiz  = upper_arm.copy(); ua_horiz[2] = 0.0
        pan_raw   = np.arctan2(ua_horiz[1], ua_horiz[0])
        joint1    = float(np.clip(pan_raw, _J_RANGE[0, 0], _J_RANGE[0, 1]))
        ua_norm   = np.linalg.norm(upper_arm) + 1e-6
        lift_raw  = np.arcsin(np.clip(-upper_arm[2] / ua_norm, -1.0, 1.0))
        joint2    = float(np.clip(
            lift_raw / self._SHOULDER_LIFT_MAX * _J_RANGE[1, 1],
            _J_RANGE[1, 0], _J_RANGE[1, 1]))
        joint3    = HOME[2]
        j4_range  = _J_RANGE[3, 1] - _J_RANGE[3, 0]
        joint4    = float(np.clip(
            _J_RANGE[3, 1] - (self._bend(upper_arm, forearm) / self._ELBOW_BEND_MAX) * j4_range,
            _J_RANGE[3, 0], _J_RANGE[3, 1]))
        joint5    = HOME[4]
        joint6    = float(np.clip(
            HOME[5] + (self._bend(forearm, hand_dir) / self._WRIST_BEND_MAX) * 1.5,
            _J_RANGE[5, 0], _J_RANGE[5, 1]))
        joint7    = HOME[6]
        return np.array([joint1, joint2, joint3, joint4, joint5, joint6, joint7])


# Alias so any old code that still imports the original name keeps working
FrankaIKRetargeter = _LegacyFrankaIKRetargeter
