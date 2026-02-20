import cv2
from camera import ZEDCamera
from detectors import StereoHandTracker
import geometry

def main():
    zed = ZEDCamera(camera_id=1) # Try 0 if 1 fails
    tracker = StereoHandTracker()

    print("Running Stereo Triangulation... Press 'q' to quit.")

    while True:
        frame_l, frame_r = zed.get_frames()
        if frame_l is None: break

        h, w, _ = frame_l.shape # Get dimensions for pixel math

        # 1. Run AI
        res_l, res_r = tracker.process(frame_l, frame_r)

        # Draw raw skeletons
        tracker.draw_landmarks(frame_l, res_l)
        tracker.draw_landmarks(frame_r, res_r)

        # 2. Extract Data (If BOTH eyes see a hand)
        if res_l.multi_hand_landmarks and res_r.multi_hand_landmarks:
            
            # Get the first hand detected in each eye
            hand_l = res_l.multi_hand_landmarks[0]
            hand_r = res_r.multi_hand_landmarks[0]

            # Get Index Finger Tip (Landmark 8)
            tip_l = hand_l.landmark[8]
            tip_r = hand_r.landmark[8]

            # Convert normalized (0.0 - 1.0) to actual Pixels
            px_l = int(tip_l.x * w)
            py_l = int(tip_l.y * h)
            
            px_r = int(tip_r.x * w)
            py_r = int(tip_r.y * h)

            # 3. Apply the Geometry (The Paper's Core Concept)
            is_valid, error = geometry.check_epipolar_constraint(py_l, py_r)

            if is_valid:
                depth = geometry.triangulate_depth(px_l, px_r)
                if depth:
                    # Success! Draw a green circle and show the depth
                    cv2.circle(frame_l, (px_l, py_l), 10, (0, 255, 0), -1)
                    cv2.putText(frame_l, f"Depth: {depth:.1f} cm", (px_l - 50, py_l - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Failure! MediaPipe hallucinated. Reject the frame.
                cv2.circle(frame_l, (px_l, py_l), 10, (0, 0, 255), -1)
                cv2.putText(frame_l, f"REJECTED (Error: {error}px)", (px_l - 50, py_l - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display
        cv2.imshow("Left Eye (Robot Perspective)", frame_l)
        # We don't strictly need to show the right eye anymore, but you can if you want
        cv2.imshow("Right Eye", frame_r) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()