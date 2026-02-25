import mujoco
import mujoco.viewer

class LeapSimulator:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def set_hand_pose(self, pos, quat):
        """
        Move the hand's mocap anchor to a new position and orientation.
        pos  : [x, y, z] in meters
        quat : [w, x, y, z] unit quaternion
        """
        # 1. Move the mocap body (what the weld constraint targets)
        self.data.mocap_pos[0]  = pos
        self.data.mocap_quat[0] = quat

        # 2. Also teleport the palm freejoint qpos to the same place.
        #    Without this, the palm starts far away and falls before
        #    the weld constraint can catch it.
        joint_id = self.model.joint("palm_free").id
        addr     = self.model.jnt_qposadr[joint_id]
        self.data.qpos[addr:addr+3] = pos        # x, y, z
        self.data.qpos[addr+3:addr+7] = quat     # w, x, y, z

        # 3. Recompute forward kinematics so the change takes effect immediately
        mujoco.mj_forward(self.model, self.data)

    def set_actuators(self, angles):
        for i, angle in enumerate(angles):
            if i < self.model.nu:
                self.data.ctrl[i] = angle

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
