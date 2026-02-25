import cv2
import mediapipe as mp

class StereoHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create TWO independent trackers.
        # model_complexity=0 is faster (~2ms less per frame); use 1 for better
        # z estimates if you have CPU headroom.
        self.tracker_left = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        self.tracker_right = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5
        )

    def process(self, frame_left, frame_right):
        """
        Runs inference on both eyes. Returns the results.
        """
        # Convert BGR to RGB
        rgb_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

        # Run Inference
        res_left = self.tracker_left.process(rgb_left)
        res_right = self.tracker_right.process(rgb_right)

        return res_left, res_right

    def draw_landmarks(self, frame, results):
        """
        Helper to draw the skeleton on a frame
        """
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                )


class ArmTracker:
    """
    MediaPipe Pose tracker for single-camera arm teleoperation.

    Detects upper-body pose and exposes shoulder / elbow / wrist landmarks
    needed to drive the 7-DOF Franka arm.

    MediaPipe Pose landmark indices (relevant subset):
      SHOULDER_L = 11    SHOULDER_R = 12
      ELBOW_L    = 13    ELBOW_R    = 14
      WRIST_L    = 15    WRIST_R    = 16
      PINKY_L    = 17    PINKY_R    = 18
      INDEX_L    = 19    INDEX_R    = 20
      THUMB_L    = 21    THUMB_R    = 22

    Coordinates are normalised image space (x, y ∈ [0,1]) plus an estimated
    z depth (relative, same scale as x/y).  world_landmarks give metric
    hip-centred coords when process() is called with enable_segmentation=False
    and model_complexity=1.
    """
    # Landmark index constants for easy reference
    SHOULDER_L = 11; SHOULDER_R = 12
    ELBOW_L    = 13; ELBOW_R    = 14
    WRIST_L    = 15; WRIST_R    = 16
    PINKY_L    = 17; PINKY_R    = 18
    INDEX_L    = 19; INDEX_R    = 20
    THUMB_L    = 21; THUMB_R    = 22

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._draw    = mp.solutions.drawing_utils
        self._pose    = self._mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def process(self, frame):
        """
        Run pose inference on a single BGR frame.

        Returns the MediaPipe PoseResult.  Access landmarks via:
            result.pose_landmarks.landmark[i]   — normalised (x, y, z)
            result.pose_world_landmarks.landmark[i] — metric, hip-centred (x, y, z)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._pose.process(rgb)

    def draw_landmarks(self, frame, result):
        """Overlay pose skeleton on frame in-place."""
        if result.pose_landmarks:
            self._draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
            )

    def close(self):
        """Release MediaPipe resources."""
        self._pose.close()