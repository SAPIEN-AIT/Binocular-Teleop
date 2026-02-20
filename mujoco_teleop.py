import time
import numpy as np
import mujoco
from mujoco import viewer

# Import your modules
from camera import ZEDCamera
from detectors import StereoHandTracker
import geometry

# --- XML DEFINITION ---
xml_string = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".8 .9 .8 1"/>
    
    <body name="hand_proxy" mocap="true" pos="0 0 0.5">
        <geom type="box" size=".05 .05 .05" rgba="1 0 0 1"/>
    </body>
    
    <body name="box1" pos="0.2 0 0.5">
        <joint type="free"/>
        <geom type="box" size=".05 .05 .05" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Global references
zed = None
tracker = None
mocap_id = None

def update_physics(model, data):
    """
    Runs every physics step.
    We process vision here but DO NOT show a window.
    """
    global zed, tracker, mocap_id
    
    if zed:
        frame_l, frame_r = zed.get_frames()
        if frame_l is not None:
            res_l, res_r = tracker.process(frame_l, frame_r)
            
            if res_l.multi_hand_landmarks and res_r.multi_hand_landmarks:
                # Use Knuckle (Index Finger MCP) for stability
                kp_l = res_l.multi_hand_landmarks[0].landmark[5]
                kp_r = res_r.multi_hand_landmarks[0].landmark[5]
                h, w, _ = frame_l.shape
                
                px_l, py_l = int(kp_l.x * w), int(kp_l.y * h)
                px_r = int(kp_r.x * w)
                
                z_cm = geometry.triangulate_depth(px_l, px_r)
                
                if z_cm:
                    # Map Real World (cm) to Sim World (meters)
                    z_m = z_cm / 100.0
                    x_norm = (kp_l.x - 0.5)
                    y_norm = (kp_l.y - 0.5)
                    
                    # Coordinate Mapping:
                    # Screen X  -> Sim X (Right)
                    # Screen Y  -> Sim Z (Up) - Inverted
                    # Screen Z  -> Sim Y (Forward) - Inverted
                    
                    sim_x = x_norm * 1.5
                    sim_y = (1.0 - z_m) + 0.5 
                    sim_z = -y_norm * 1.0 + 0.5 
                    
                    print(f"Hand Depth: {z_cm:.1f}cm -> Sim Pos: [{sim_x:.2f}, {sim_y:.2f}, {sim_z:.2f}]")
    
                    data.mocap_pos[mocap_id] = [sim_x, sim_y, sim_z]

def main():
    global zed, tracker, mocap_id
    
    zed = ZEDCamera(camera_id=1) # Try 0 if 1 fails
    tracker = StereoHandTracker()
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mocap_id = model.body("hand_proxy").mocapid[0]

    print("Starting Simulation... Move your hand to move the Red Cube.")
    print("Press ESC in the viewer window to quit.")

    # Launch the viewer (Blocking)
    # This will open the MuJoCo window and call update_physics repeatedly
    viewer.launch(model, data, loader=None)

    zed.close()

if __name__ == "__main__":
    main()