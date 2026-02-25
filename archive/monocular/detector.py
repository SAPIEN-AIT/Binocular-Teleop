import cv2
import mediapipe as mp 

class HandDetector: 
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.tracker = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.tracker.process(rgb)

    def get_landmarks(self, results):
        if not results.multi_hand_landmarks:
            return None
        return results.multi_hand_landmarks[0].landmark

    def draw(self, frame, results): 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )