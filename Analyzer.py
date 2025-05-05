import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import face_recognition
import io
import torch

class FaceAnalyzer:
    FRAME_SKIP = 3
    MAX_FACES = 3
    FACE_MATCH_THRESHOLD = 0.5
    SPEAKING_LIP_DIST_THRESHOLD = 0.03 

    def __init__(self, model_path, temp_faces=None):
        self.model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True)
        self.known_faces = self._load_temp_faces(temp_faces) if temp_faces else {'encodings': [], 'names': []}
        
        self.frame_counter = 0
        self.tracked_faces = {}
        self.next_id = 1
        self.speaking_times = {}
        self.active_speakers = {}
        
        self.lip_indices = list(range(61, 69)) + list(range(291, 299))
        self.eyebrow_indices = {'left': [70, 63, 105, 66, 107], 'right': [336, 296, 334, 300, 293]}  
        self.eye_indices = {'left': [33, 160, 158, 133], 'right': [362, 385, 387, 263]}  

        self.show_names = True
        self.show_times = True
        self.show_emotion = True
        self.show_bounding_boxes = True
        self.fps = 30  # Varsayılan FPS

    def set_fps(self, fps):
        """Dışarıdan FPS güncellemesi için yardımcı fonksiyon."""
        if fps > 0:
            self.fps = fps

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
        
        if self.frame_counter % self.FRAME_SKIP == 0:
            self._detect_faces(rgb_frame)
        
        # Reset active_speakers for this frame
        self.active_speakers = {}
        self._process_face_landmarks(frame, rgb_frame)
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
        cv2.putText(frame, emotion, (cx-50, cy-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _update_speaking_status(self, landmarks, w, h):
        lm_coords = landmarks.landmark

        face_height = abs(lm_coords[152].y - lm_coords[10].y)

        upper_lip_idx = 13  
        lower_lip_idx = 14  
        upper_lip_y = lm_coords[upper_lip_idx].y
        lower_lip_y = lm_coords[lower_lip_idx].y

        lip_dist = abs(upper_lip_y - lower_lip_y) / face_height

        landmark_center = (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h))

        # En yakın yüzü bul ve sadece ona konuşma durumu ata
        min_dist = float('inf')
        closest_name = None
        for _, data in self.tracked_faces.items():
            if 'center' not in data:
                continue
            dist = np.sqrt((landmark_center[0] - data['center'][0])**2 + 
                           (landmark_center[1] - data['center'][1])**2)
            if dist < min_dist and dist < 50:
                min_dist = dist
                closest_name = data['name']

        if closest_name is not None:
            is_speaking = lip_dist > self.SPEAKING_LIP_DIST_THRESHOLD
            self.active_speakers[closest_name] = is_speaking
            if is_speaking:
                self.speaking_times[closest_name] = self.speaking_times.get(closest_name, 0) + (self.FRAME_SKIP / self.fps)

    def _is_face_near_landmark(self, face_data, landmark_center):
        if 'center' not in face_data:
            return False
        dist = np.sqrt((landmark_center[0] - face_data['center'][0])**2 + 
                      (landmark_center[1] - face_data['center'][1])**2)
        return dist < 50

    def _draw_face_info(self, frame):
        for _, data in self.tracked_faces.items():
            if 'box' not in data:
                continue
                
            x1, y1, x2, y2 = data['box']
            
            if self.show_bounding_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            name = data['name']
            duration = self.speaking_times.get(name, 0)
            status = "Speaking" if self.active_speakers.get(name, False) else ""
            
            text_to_display = []
            if self.show_names:
                text_to_display.append(name)
            if self.show_times:
                text_to_display.append(f"{duration:.1f}s")
            if status:
                text_to_display.append(status)
                
            if text_to_display:
                display_text = " ".join(text_to_display)
                
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y2 + 5), (x1 + text_size[0], y2 + 25), (0, 0, 0, 128), -1)
                cv2.putText(frame, display_text, (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
            if dist < min_dist:
                min_dist = dist
                closest_id = fid
                
        if min_dist < 75: 
            return closest_id
        return None

    def _add_new_face(self, rgb_frame, box, center):
        x1, y1, x2, y2 = box
        face_location = (y1, x2, y2, x1)
        name = "Unknown"
        
        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        if encodings:
            name = self._identify_face(encodings[0])
        if name == "Unknown":
            return  
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
        face_height = abs(lm_coords[152].y - lm_coords[10].y)
        lip_dist = self._calculate_lip_distance(lm_coords, face_height)
        eye_avg = self._calculate_eye_openness(lm_coords, face_height)
        eyebrow_avg = self._calculate_eyebrow_position(lm_coords)
        return self._determine_emotion(lip_dist, eye_avg, eyebrow_avg)

    def _calculate_lip_distance(self, lm_coords, face_height):
        upper_lip_points = [13, 312, 82, 191]
        lower_lip_points = [14, 87, 317, 375]
        upper_lip_y = np.mean([lm_coords[i].y for i in upper_lip_points])
        lower_lip_y = np.mean([lm_coords[i].y for i in lower_lip_points])
        return abs(upper_lip_y - lower_lip_y) / face_height

    def _calculate_eye_openness(self, lm_coords, face_height):
        left_eye_h = abs(lm_coords[159].y - lm_coords[145].y) / face_height
        right_eye_h = abs(lm_coords[386].y - lm_coords[374].y) / face_height
        return (left_eye_h + right_eye_h) / 2

    def _calculate_eyebrow_position(self, lm_coords):
        left_eyebrow = np.mean([lm_coords[i].y for i in self.eyebrow_indices['left']])
        left_eye = np.mean([lm_coords[i].y for i in self.eye_indices['left']])
        right_eyebrow = np.mean([lm_coords[i].y for i in self.eyebrow_indices['right']])
        right_eye = np.mean([lm_coords[i].y for i in self.eye_indices['right']])
        left_dist = left_eyebrow - left_eye
        right_dist = right_eyebrow - right_eye
        return (left_dist + right_dist) / 2

    def _determine_emotion(self, lip_dist, eye_avg, eyebrow_dist):
        if not self.show_emotion:
            return ("", (0, 0, 0))
        if lip_dist > 0.08 and eye_avg > 0.045 and eyebrow_dist > 0.03:
            return ("SURPRISED", (255, 255, 0))
        elif lip_dist > 0.06 and eye_avg > 0.035:
            return ("HAPPY", (0, 255, 0))
        elif eyebrow_dist < 0.01 and eye_avg < 0.025:
            return ("ANNOYED", (0, 0, 255))
        else:
            return ("NEUTRAL", (200, 200, 200))