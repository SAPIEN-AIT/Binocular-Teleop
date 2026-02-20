import cv2
import mediapipe as mp

class StereoHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create TWO independent trackers
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