import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import face_recognition
import io

class FaceAnalyzer:
    def __init__(self, model_path="src/yolov11l-face.pt", temp_faces=None):
        self.model = YOLO(model_path).to("cuda")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True)
        self.known_faces = self._load_temp_faces(temp_faces) if temp_faces else {'encodings': [], 'names': []}
        
        self.frame_counter = 0
        self.tracked_faces = {}
        self.next_id = 1
        self.speaking_times = {}
        self.active_speakers = {}
        
        self.FRAME_SKIP = 2
        self.MAX_FACES = 5
        self.FACE_MATCH_THRESHOLD = 0.6
        
        self.lip_indices = list(range(61, 69)) + list(range(291, 299))
        self.eyebrow_indices = {'left': [70, 63, 105], 'right': [336, 296, 334]}

    def _load_temp_faces(self, temp_faces):
        known_data = {'encodings': [], 'names': []}
        try:
            for name, image_data in temp_faces.items():
                img = face_recognition.load_image_file(io.BytesIO(image_data))
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_data['encodings'].append(encodings[0])
                    known_data['names'].append(name)
        except Exception as e:
            print(f"Temp face loading error: {e}")
        return known_data

    def process_frame(self, frame):
        self.frame_counter += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        if self.frame_counter % self.FRAME_SKIP == 0:
            results = self.model(rgb_frame, verbose=False)[0]
            boxes = [tuple(map(int, box.xyxy[0].tolist())) for box in results.boxes]
            self._update_tracking(boxes, rgb_frame)
        
        if mesh_results := self.face_mesh.process(rgb_frame).multi_face_landmarks:
                for landmarks in mesh_results:
                    emotion, color = self._analyze_emotion(landmarks)
                    cx = int(landmarks.landmark[1].x * w)
                    cy = int(landmarks.landmark[1].y * h)
                    cv2.putText(frame, f"Emotion: {emotion}", (cx-50, cy-50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        for face_id, data in self.tracked_faces.items():
            if 'box' in data:
                x1, y1, x2, y2 = data['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                name = data['name']
                duration = self.speaking_times.get(name, 0)
                cv2.putText(frame, f"{name} ({duration:.1f}s)", (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

    def _update_tracking(self, boxes, rgb_frame):
        for fid in self.tracked_faces:
            self.tracked_faces[fid]['active'] = False
            
        for box in boxes[:self.MAX_FACES]:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            face_id = None
            min_dist = float('inf')
            
            for fid, data in self.tracked_faces.items():
                if 'center' in data:
                    dist = np.sqrt((center[0] - data['center'][0])**2 + 
                                   (center[1] - data['center'][1])**2)
                    if dist < min_dist and dist < 100:
                        min_dist = dist
                        face_id = fid
            
            if face_id:  # Update existing face
                self.tracked_faces[face_id].update({'center': center, 'box': box, 'active': True})
            else:  # Create new face
                face_location = (y1, x2, y2, x1)
                name = "Unknown"
                
                if encodings := face_recognition.face_encodings(rgb_frame, [face_location]):
                    name = self._identify_face(encodings[0])
                
                self.tracked_faces[self.next_id] = {
                    'name': name, 'center': center, 'box': box, 'active': True
                }
                self.next_id += 1
        
        self.tracked_faces = {fid: data for fid, data in self.tracked_faces.items() if data['active']}

    def _identify_face(self, encoding):
        if not self.known_faces['encodings']:
            return "Unknown"
            
        distances = face_recognition.face_distance(self.known_faces['encodings'], encoding)
        if len(distances) > 0:
            best_match = np.argmin(distances)
            if distances[best_match] < self.FACE_MATCH_THRESHOLD:
                return self.known_faces['names'][best_match]
        
        return "Unknown"

    def _analyze_emotion(self, landmarks):
        # Cache landmark coordinates for faster access
        lm_coords = landmarks.landmark
        
        # Calculate lip distance more efficiently
        upper_lip_y = sum(lm_coords[i].y for i in self.lip_indices[:8]) / 8
        lower_lip_y = sum(lm_coords[i].y for i in self.lip_indices[8:]) / 8
        lip_dist = abs(upper_lip_y - lower_lip_y)

        # Calculate eyebrow position directly
        left_eyebrow_y = sum(lm_coords[i].y for i in self.eyebrow_indices['left']) / len(self.eyebrow_indices['left'])
        right_eyebrow_y = sum(lm_coords[i].y for i in self.eyebrow_indices['right']) / len(self.eyebrow_indices['right'])
        eyebrow_avg = (left_eyebrow_y + right_eyebrow_y) * 0.5  # Faster than division by 2.0
        
        # Use predefined emotion tuples to avoid repeated tuple creation
        HAPPY = ("HAPPY", (0, 255, 0))
        ANNOYED = ("ANNOYED", (0, 0, 255))
        NEUTRAL = ("NEUTRAL", (200, 200, 200))
        
        # Determine emotion with early returns
        if lip_dist > 0.04:
            return HAPPY
        if eyebrow_avg < 0.27:
            return ANNOYED
        return NEUTRAL



def main():
    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera connection error")
        return

    cv2.namedWindow("Face Analysis", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = analyzer.process_frame(frame)
            cv2.imshow("Face Analysis", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        analyzer.print_results()

if __name__ == "__main__":
    main()
