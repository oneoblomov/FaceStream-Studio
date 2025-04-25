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
        self.eyebrow_indices = {'left': [70, 63, 105, 66, 107], 'right': [336, 296, 334, 300, 293]}  
        self.eye_indices = {'left': [33, 160, 158, 133], 'right': [362, 385, 387, 263]}  

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
        
        # Process with model if needed
        if self.frame_counter % self.FRAME_SKIP == 0:
            self._detect_faces(rgb_frame)
            
        # Process face landmarks
        self._process_face_landmarks(frame, rgb_frame)
        
        # Draw face boxes and info
        self._draw_face_info(frame)
        
        return frame

    def _detect_faces(self, rgb_frame):
        results = self.model(rgb_frame, verbose=False)[0]
        boxes = [tuple(map(int, box.xyxy[0].tolist())) for box in results.boxes]
        self._update_tracking(boxes, rgb_frame)

    def _process_face_landmarks(self, frame, rgb_frame):
        mesh_results = self.face_mesh.process(rgb_frame).multi_face_landmarks
        if not mesh_results:
            return
            
        h, w = frame.shape[:2]
        for landmarks in mesh_results:
            emotion, color = self._analyze_emotion(landmarks)
            self._display_emotion(frame, landmarks, emotion, color, w, h)
            self._update_speaking_status(landmarks, w, h)

    def _display_emotion(self, frame, landmarks, emotion, color, w, h):
        cx = int(landmarks.landmark[1].x * w)
        cy = int(landmarks.landmark[1].y * h)
        cv2.putText(frame, f"Emotion: {emotion}", (cx-50, cy-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _update_speaking_status(self, landmarks, w, h):
        lm_coords = landmarks.landmark
        upper_lip_y = sum(lm_coords[i].y for i in self.lip_indices[:8]) / 8
        lower_lip_y = sum(lm_coords[i].y for i in self.lip_indices[8:]) / 8
        lip_dist = abs(upper_lip_y - lower_lip_y)
        
        landmark_center = (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h))
        for face_id, data in self.tracked_faces.items():
            if not self._is_face_near_landmark(data, landmark_center):
                continue
                
            name = data['name']
            is_speaking = lip_dist > 0.025
            self.active_speakers[name] = is_speaking
            
            if is_speaking:
                self.speaking_times[name] = self.speaking_times.get(name, 0) + 1/30
            break

    def _is_face_near_landmark(self, face_data, landmark_center):
        if 'center' not in face_data:
            return False
        dist = np.sqrt((landmark_center[0] - face_data['center'][0])**2 + 
                      (landmark_center[1] - face_data['center'][1])**2)
        return dist < 50

    def _draw_face_info(self, frame):
        for face_id, data in self.tracked_faces.items():
            if 'box' not in data:
                continue
                
            x1, y1, x2, y2 = data['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            name = data['name']
            duration = self.speaking_times.get(name, 0)
            status = "Speaking" if self.active_speakers.get(name, False) else ""
            cv2.putText(frame, f"{name} ({duration:.1f}s) {status}", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def print_results(self):
        print("\n--- Speaking Time Results ---")
        for name, duration in self.speaking_times.items():
            print(f"{name}: {duration:.1f} seconds")
        print("---------------------------\n")

    def _update_tracking(self, boxes, rgb_frame):
        self._mark_faces_inactive()
        
        for box in boxes[:self.MAX_FACES]:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            face_id = self._find_closest_face(center)
            
            if face_id:
                self.tracked_faces[face_id].update({'center': center, 'box': box, 'active': True})
            else:
                self._add_new_face(rgb_frame, box, center)
        
        self._remove_inactive_faces()

    def _mark_faces_inactive(self):
        for fid in self.tracked_faces:
            self.tracked_faces[fid]['active'] = False

    def _find_closest_face(self, center):
        min_dist = float('inf')
        closest_id = None
        
        for fid, data in self.tracked_faces.items():
            if 'center' not in data:
                continue
                
            dist = np.sqrt((center[0] - data['center'][0])**2 + 
                          (center[1] - data['center'][1])**2)
            if dist < min_dist and dist < 100:
                min_dist = dist
                closest_id = fid
                
        return closest_id

    def _add_new_face(self, rgb_frame, box, center):
        x1, y1, x2, y2 = box
        face_location = (y1, x2, y2, x1)
        name = "Unknown"
        
        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        if encodings:
            name = self._identify_face(encodings[0])
        
        self.tracked_faces[self.next_id] = {
            'name': name, 'center': center, 'box': box, 'active': True
        }
        self.next_id += 1

    def _remove_inactive_faces(self):
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
        lm_coords = landmarks.landmark
        
        face_width = abs(lm_coords[454].x - lm_coords[234].x)
        face_height = abs(lm_coords[152].y - lm_coords[10].y)
        
        lip_dist = self._calculate_lip_distance(lm_coords, face_height)
        eye_avg = self._calculate_eye_openness(lm_coords, face_height)
        eyebrow_avg = self._calculate_eyebrow_position(lm_coords)
        
        return self._determine_emotion(lip_dist, eye_avg, eyebrow_avg)

    def _calculate_lip_distance(self, lm_coords, face_height):
        upper_lip_y = sum(lm_coords[i].y for i in self.lip_indices[:8]) / 8
        lower_lip_y = sum(lm_coords[i].y for i in self.lip_indices[8:]) / 8
        return abs(upper_lip_y - lower_lip_y) / face_height

    def _calculate_eye_openness(self, lm_coords, face_height):
        left_eye_h = abs(lm_coords[159].y - lm_coords[145].y) / face_height
        right_eye_h = abs(lm_coords[386].y - lm_coords[374].y) / face_height
        return (left_eye_h + right_eye_h) / 2

    def _calculate_eyebrow_position(self, lm_coords):
        left_eyebrow_y = sum(lm_coords[i].y for i in self.eyebrow_indices['left']) / len(self.eyebrow_indices['left'])
        right_eyebrow_y = sum(lm_coords[i].y for i in self.eyebrow_indices['right']) / len(self.eyebrow_indices['right'])
        return (left_eyebrow_y + right_eyebrow_y) * 0.5

    def _determine_emotion(self, lip_dist, eye_avg, eyebrow_avg):
        if lip_dist > 0.1 and eye_avg > 0.05:
            return ("HAPPY", (0, 255, 0))
        elif eyebrow_avg < 0.25 and eye_avg < 0.03:
            return ("ANNOYED", (0, 0, 255))
        else:
            return ("NEUTRAL", (200, 200, 200))
        
    def detect_unknown_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        unknown_faces = []
        
        for (top, right, bottom, left) in face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if not face_encoding or self._identify_face(face_encoding[0]) != "Unknown":
                continue
            unknown_faces.append(frame[top:bottom, left:right])
        
        return unknown_faces
