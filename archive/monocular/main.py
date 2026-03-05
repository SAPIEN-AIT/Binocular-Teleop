from camera import MonocularCamera
from detector import HandDetector
from retargeting import landmarks_to_joints, wrist_to_world_pos
from sim import LeapSimulator

XML_PATH = "scene.xml"

def main():
    camera = MonocularCamera(camera_id=0)
    detector = HandDetector()
    sim = LeapSimulator(XML_PATH)
    sim.launch_viewer()

    # Set initial hand position: (x, y, z) in meters, quat (w,x,y,z)
    # Lower z = hand closer to floor. Tune this until it's above the box.
    sim.set_hand_pose(pos=[0, 0, 0.18], quat=[1, 0, 0, 0])

    print("Running — close the MuJoCo window to quit.")

    while sim.viewer.is_running():
        frame = camera.get_frame()
        if frame is None:
            continue

        results = detector.process(frame)
        landmarks = detector.get_landmarks(results)

        if landmarks is not None:
            # Move the hand to follow the real wrist position (2D plane)
            pos = wrist_to_world_pos(landmarks, z_fixed=0.18)
            sim.data.mocap_pos[0] = pos

            # Drive the finger joints from landmark angles
            joints = landmarks_to_joints(landmarks)
            sim.set_actuators(joints)

        sim.step()

    camera.close()
    sim.close()

if __name__ == "__main__":
    main()
