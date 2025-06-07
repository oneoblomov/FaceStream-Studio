import io
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
import mediapipe as mp
from ultralytics import YOLO
import face_recognition
class EmotionMLP(nn.Module):
    def __init__(self, input_dim=936, num_classes=7):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.25)
        )
        self.middle_main = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.middle_shortcut = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.initial(x)
        identity = x
        x = self.middle_main(x)
        x = x + self.middle_shortcut(identity)
        x = self.final(x)
        return x

class FaceAnalyzer:
    EMOTION_COLOR = (255, 255, 255)
    MAX_FACE_ID_CACHE_SIZE = 128
    MAX_EMOTION_CACHE_SIZE = 100
    
    def __init__(self, model_path, temp_faces=None, fps=30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        
        self.FRAME_SKIP = 2          
        self.SPEAKING_LIP_DIST_THRESHOLD = 0.07
        self.FACE_MATCH_THRESHOLD = 0.6
        self.MAX_FACES = 10

        self.face_mesh = mp.solutions.face_mesh.FaceMesh( # type: ignore
            max_num_faces=self.MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        self.known_faces = self._load_temp_faces(temp_faces) if temp_faces else {'encodings': [], 'names': []}
        
        self.frame_counter = 0
        self.tracked_faces = {}
        self.next_id = 1
        self.speaking_times = {}
        self.active_speakers = {}
        self.emotion_cache = {}
        
        self.show_names = True
        self.show_times = True
        self.show_emotion = True
        self.show_bounding_boxes = True
        self.fps = fps
        
        self._load_emotion_models()
        self.face_id_cache = {} 
    
    def _load_emotion_models(self):
        """Lazily load emotion models when needed"""
        model_dir = os.path.dirname(__file__)
        self.torch_model = None
        self.torch_scaler = None
        self.torch_le = None
        self.using_torch = False
        try:
            torch_model_path = os.path.join(model_dir, "src", "models", "torch", "emotion_mlp.pth")
            torch_scaler_path = os.path.join(model_dir, "src", "models", "torch", "emotion_scaler.pkl")
            torch_le_path = os.path.join(model_dir, "src", "models", "torch", "emotion_labelencoder.pkl")
            if all(os.path.exists(p) for p in [torch_model_path, torch_scaler_path, torch_le_path]):
                self.torch_model = EmotionMLP(input_dim=936, num_classes=7).to(self.device)
                self.torch_model.load_state_dict(torch.load(torch_model_path, map_location=self.device))
                self.torch_model.eval()
                self.torch_scaler = joblib.load(torch_scaler_path)
                self.torch_le = joblib.load(torch_le_path)
                self.using_torch = True
                print("PyTorch duygu sınıflandırma modeli başarıyla yüklendi.")
                return
        except Exception as e:
            print(f"PyTorch model yüklenemedi: {e}")

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
        
        self.active_speakers = {}
        self._process_face_landmarks(frame, rgb_frame)
        self._draw_face_info(frame)
        
        return frame

    def _detect_faces(self, rgb_frame):
        if not self.show_bounding_boxes and not self.show_names:
            return
            
        results = self.model(rgb_frame, verbose=False)[0]
        boxes = [tuple(map(int, box.xyxy[0].tolist())) for box in results.boxes]
        self._update_tracking(boxes, rgb_frame)

    def _process_face_landmarks(self, frame, rgb_frame):
        mesh_results = self.face_mesh.process(rgb_frame).multi_face_landmarks
        if not mesh_results:
            return
            
        h, w = frame.shape[:2]
        for landmarks in mesh_results:
            if self.show_emotion:
                landmark_id = self._get_landmark_id(landmarks, w, h)
                emotion, color = self._analyze_emotion(landmarks, landmark_id)
                self._display_emotion(frame, landmarks, emotion, color, w, h)
            self._update_speaking_status(landmarks, w, h)
    
    def _get_landmark_id(self, landmarks, w, h):
        """Create a unique ID for this landmark to use in caching"""
        cx = int(landmarks.landmark[1].x * w)
        cy = int(landmarks.landmark[1].y * h)
        return f"{cx}_{cy}"

    def _display_emotion(self, frame, landmarks, emotion, color, w, h):
        if not emotion:
            return
        cx = int(landmarks.landmark[1].x * w)
        cy = int(landmarks.landmark[1].y * h)
        cv2.putText(frame, emotion, (cx-50, cy-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _update_speaking_status(self, landmarks, w, h):
        lm_coords = landmarks.landmark

        face_height = abs(lm_coords[152].y - lm_coords[10].y)
        lip_dist = abs(lm_coords[13].y - lm_coords[14].y) / face_height

        landmark_center = (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h))

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
                self.speaking_times[closest_name] = self.speaking_times.get(closest_name, 0) + (1.0 / self.fps)

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
            
            text_parts = []
            if self.show_names:
                text_parts.append(name)
            if self.show_times:
                text_parts.append(f"{duration:.1f}s")
            if status:
                text_parts.append(status)
                
            if text_parts:
                display_text = " ".join(text_parts)
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                rect_y1 = y2 + 5
                rect_y2 = min(y2 + 25, frame.shape[0]-1)
                rect_x2 = min(x1 + text_size[0], frame.shape[1]-1)
                
                cv2.rectangle(frame, (x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                cv2.putText(frame, display_text, (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _update_tracking(self, boxes, rgb_frame):
        for fid in self.tracked_faces:
            self.tracked_faces[fid]['active'] = False
        
        for box in boxes[:self.MAX_FACES]:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            face_id = self._find_closest_face(center)
            
            if face_id:
                self.tracked_faces[face_id].update({'center': center, 'box': box, 'active': True})
            else:
                self._add_new_face(rgb_frame, box, center)
        
        self.tracked_faces = {fid: data for fid, data in self.tracked_faces.items() if data['active']}

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
                
        return closest_id if min_dist < 75 else None

    def _add_new_face(self, rgb_frame, box, center):
        x1, y1, x2, y2 = box
        face_location = (min(y1, y2), max(x1, x2), max(y1, y2), min(x1, x2))
        
        if not self.known_faces['encodings']:
            return
            
        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        if encodings:
            encoding = encodings[0]
            name = self._identify_face(encoding)
            if name != "Unknown":
                self.tracked_faces[self.next_id] = {
                    'name': name, 'center': center, 'box': box, 'active': True
                }
                self.next_id += 1

    def _manage_cache(self, cache, key, value, max_size):
        """Centralized cache management to avoid code duplication"""
        cache[key] = value
        if len(cache) > max_size:
            cache.pop(next(iter(cache)))

    def _identify_face(self, encoding):
        """Identify a face using a manual cache for frequently seen faces"""
        encoding_tuple = tuple(np.round(encoding, 5))

        if encoding_tuple in self.face_id_cache:
            return self.face_id_cache[encoding_tuple]

        if not self.known_faces['encodings']:
            return "Unknown"
            
        distances = face_recognition.face_distance(self.known_faces['encodings'], encoding)
        if len(distances) > 0:
            best_match = np.argmin(distances)
            if distances[best_match] < self.FACE_MATCH_THRESHOLD:
                name = self.known_faces['names'][best_match]
                self._manage_cache(self.face_id_cache, encoding_tuple, name, self.MAX_FACE_ID_CACHE_SIZE)
                return name
        
        self._manage_cache(self.face_id_cache, encoding_tuple, "Unknown", self.MAX_FACE_ID_CACHE_SIZE)
        return "Unknown"

    def _analyze_emotion(self, landmarks, landmark_id=None):
        """Face expression based emotion recognition with caching"""
        if not self.show_emotion:
            return ("", self.EMOTION_COLOR)
        if landmark_id and landmark_id in self.emotion_cache:
            return self.emotion_cache[landmark_id]
        result = ("UNKNOWN", self.EMOTION_COLOR)
        if self.using_torch and self.torch_model:
            try:
                result = self._predict_emotion_torch(landmarks)
            except Exception as e:
                print(f"PyTorch emotion prediction failed: {e}")
        if landmark_id:
            self._manage_cache(self.emotion_cache, landmark_id, result, self.MAX_EMOTION_CACHE_SIZE)
        return result

    def _predict_emotion_torch(self, landmarks):
        """Predict emotion using PyTorch model"""
        if not self.torch_scaler or not self.torch_model or not self.torch_le:
            return ("UNKNOWN", self.EMOTION_COLOR)
            
        features = self._extract_landmark_features(landmarks)
        
        features_scaled = self.torch_scaler.transform([features])
        
        tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.torch_model(tensor)
            pred = torch.argmax(logits, dim=1).cpu().item()
            emotion = self.torch_le.inverse_transform([pred])[0]
        
        return (emotion.upper(), self.EMOTION_COLOR)
    
    def _extract_landmark_features(self, landmarks):
        """Extract and normalize landmark features - simplified version"""
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        features = xs + ys
        
        if len(features) != 936:
            features = (features + [0.0] * 936)[:936]
        
        return features