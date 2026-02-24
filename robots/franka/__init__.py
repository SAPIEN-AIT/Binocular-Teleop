"""
robots/franka — Franka FR3 controller (placeholder).

This package is not yet implemented.  When adding Franka control:

1. Implement ``FrankaController(RobotController)`` here with:
   - ``retarget(landmarks) -> np.ndarray``  (7 joint targets or end-effector pose)
   - ``scene_xml_path() -> str``            (path to franka_fr3_v2/scene.xml or similar)

2. Replace the LEAP-specific constants in mujoco_teleop.py (START_Y, mocap body
   name, scene XML path) with a ``robot = FrankaController(model)`` call so the
   rest of the loop stays robot-agnostic.

Reference XML: ``franka_fr3_v2/scene.xml``
"""

import os
import numpy as np
import mujoco
from robots.base import RobotController


class FrankaController(RobotController):
    """
    Placeholder: Franka FR3 controller.

    Raises NotImplementedError until the retargeting logic is written.
    """

    def retarget(self, landmarks) -> np.ndarray:
        raise NotImplementedError("Franka retargeting is not yet implemented.")

    @classmethod
    def scene_xml_path(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "franka_fr3_v2", "scene.xml",
        )
