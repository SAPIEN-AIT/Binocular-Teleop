import cv2

class ZEDCamera:
    def __init__(self, camera_id=1):
        # 1. Open Camera
        self.cap = cv2.VideoCapture(camera_id)
        
        # 2. Force Side-by-Side Resolution (2560x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            print(f"Warning: Camera {camera_id} failed to open.")

    def get_frames(self):
        """
        Returns (frame_left, frame_right)
        """
        success, frame = self.cap.read()
        if not success:
            return None, None

        # Split the image
        h, w, _ = frame.shape
        half_w = w // 2
        return frame[:, :half_w], frame[:, half_w:]

    def close(self):
        self.cap.release()