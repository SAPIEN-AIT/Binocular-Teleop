"""
robots/franka — Franka FR3 arm controller.

Exports
-------
  FrankaJacobianIK     Jacobian IK retargeter (wrist 3-D position → 7 joint angles)
  FrankaIKRetargeter   [legacy] direct angle-based retargeter (kept for reference)
  FrankaController     thin RobotController wrapper (uses FrankaJacobianIK)
"""

from robots.franka.ik_retargeting import FrankaJacobianIK   # noqa: F401
from robots.franka.ik_retargeting import FrankaIKRetargeter  # noqa: F401  (legacy alias)

import os
import numpy as np
import mujoco
from robots.base import RobotController


class FrankaController(RobotController):
    """
    RobotController implementation for the Franka FR3 arm.

    Wraps FrankaJacobianIK so the teleoperation loop can use the generic
    RobotController interface.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        self._ik = FrankaJacobianIK(model)

    def retarget(self, landmarks) -> np.ndarray:
        """Map MediaPipe Pose result → 7 Franka actuator targets via IK."""
        from robots.franka.ik_retargeting import _lm3, SHOULDER_R, WRIST_R
        if landmarks is None or landmarks.pose_world_landmarks is None:
            return self._ik.last_q()
        lm = landmarks.pose_world_landmarks.landmark
        if lm[SHOULDER_R].visibility < 0.5 or lm[WRIST_R].visibility < 0.5:
            return self._ik.last_q()
        wrist    = _lm3(lm, WRIST_R)
        shoulder = _lm3(lm, SHOULDER_R)
        return self._ik.solve(wrist, shoulder)

    @classmethod
    def scene_xml_path(cls) -> str:
        """Path to the Franka scene XML (co-located with this package)."""
        return os.path.join(os.path.dirname(__file__), "scene.xml")
