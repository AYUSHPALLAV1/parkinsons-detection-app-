import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # For iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for eyes
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate EAR to detect blinks."""
        # Vertical distances
        v1 = np.linalg.norm(np.array(landmarks[eye_indices[12]]) - np.array(landmarks[eye_indices[4]]))
        v2 = np.linalg.norm(np.array(landmarks[eye_indices[14]]) - np.array(landmarks[eye_indices[2]]))
        # Horizontal distance
        h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[8]]))
        
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def get_iris_center(self, landmarks, iris_indices):
        """Calculate the center of the iris for gaze tracking."""
        iris_pts = [landmarks[i] for i in iris_indices]
        center = np.mean(iris_pts, axis=0)
        return center

    def process_frame(self, frame):
        """Process a single frame and return landmarks and eye features."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Convert landmarks to pixel coordinates
        landmarks_px = []
        for lm in face_landmarks.landmark:
            landmarks_px.append([lm.x * w, lm.y * h, lm.z * w])
        
        # Extract features
        left_ear = self.get_eye_aspect_ratio(landmarks_px, self.LEFT_EYE)
        right_ear = self.get_eye_aspect_ratio(landmarks_px, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        left_iris_center = self.get_iris_center(landmarks_px, self.LEFT_IRIS)
        right_iris_center = self.get_iris_center(landmarks_px, self.RIGHT_IRIS)
        
        return {
            'ear': ear,
            'left_iris': left_iris_center,
            'right_iris': right_iris_center,
            'landmarks': landmarks_px
        }

def run_eye_tracking_session(duration=30):
    """Run a basic eye tracking session for testing."""
    cap = cv2.VideoCapture(0)
    tracker = EyeTracker()
    start_time = time.time()
    data = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
            
        features = tracker.process_frame(frame)
        if features:
            data.append({
                'timestamp': time.time() - start_time,
                'ear': features['ear'],
                'gaze_x': (features['left_iris'][0] + features['right_iris'][0]) / 2.0,
                'gaze_y': (features['left_iris'][1] + features['right_iris'][1]) / 2.0
            })
            
            # Visual feedback
            cv2.circle(frame, (int(features['left_iris'][0]), int(features['left_iris'][1])), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(features['right_iris'][0]), int(features['right_iris'][1])), 2, (0, 255, 0), -1)
            cv2.imshow('Eye Tracking', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return data
