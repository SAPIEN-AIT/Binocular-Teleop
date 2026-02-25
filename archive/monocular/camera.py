import cv2 

class MonocularCamera : 
    
    def __init__(self, camera_id = 0): 
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened(): 
            print("No camera found")
            
    def get_frame(self): 
        success, frame = self.cap.read()

        return frame if success else None
    
    def close(self): 
        self.cap.release()