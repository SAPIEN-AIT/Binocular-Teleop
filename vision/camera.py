import cv2
import numpy as np

class ZEDCamera:
    def __init__(self, camera_id=1, y_offset: int = 0):
        """
        Parameters
        ----------
        camera_id : int
            OpenCV device index for the ZED camera.
        y_offset : int
            Vertical pixel shift applied to the RIGHT frame to correct a
            known sensor misalignment between the two lenses.
            Positive → shifts the right frame DOWN.
            Negative → shifts the right frame UP.
            Use ``binocular.geometry.Y_OFFSET_PX`` as the canonical value.
        """
        # 1. Open Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.y_offset = y_offset

        # 2. Force Side-by-Side Resolution (2560x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            print(f"Warning: Camera {camera_id} failed to open.")

    def get_frames(self):
        """
        Returns (frame_left, frame_right) after applying vertical alignment
        correction to the right frame when y_offset != 0.
        """
        success, frame = self.cap.read()
        if not success:
            return None, None

        h, w, _ = frame.shape
        half_w = w // 2
        left  = frame[:, :half_w]
        right = frame[:, half_w:]

        # Correct the systematic vertical misalignment between the two lenses.
        # warpAffine with a pure translation is the most efficient way to do
        # this without resampling artifacts.
        if self.y_offset != 0:
            M = np.float32([[1, 0, 0], [0, 1, self.y_offset]])
            right = cv2.warpAffine(right, M, (right.shape[1], right.shape[0]))

        return left, right

    def close(self):
        self.cap.release()