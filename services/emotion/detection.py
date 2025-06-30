import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any

class FaceDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model = self._load_model(config['model_path'])
        self.min_confidence = config['min_confidence']
        self.max_faces = config['max_faces']
        self.input_size = tuple(config['input_size'])
        self.landmark_points = config['landmark_points']
        
    def _load_model(self, model_path: str) -> ort.InferenceSession:
        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        return ort.InferenceSession(
            model_path,
            providers=providers
        )
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0  # InsightFace normalization
        img = img.transpose(2, 0, 1)  # HWC to CHW
        return np.expand_dims(img, axis=0)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = frame.shape[:2]
        input_data = self._preprocess(frame)
        
        outputs = self.model.run(
            None,
            {"data": input_data}
        )
        
        bboxes = outputs[0]
        landmarks = outputs[1]
        scores = outputs[2]
        
        faces = []
        for i in range(min(bboxes.shape[1], self.max_faces)):
            confidence = float(scores[0, i])
            if confidence < self.min_confidence:
                continue
                
            box = bboxes[0, i]
            x1 = int(box[0] * orig_w)
            y1 = int(box[1] * orig_h)
            x2 = int(box[2] * orig_w)
            y2 = int(box[3] * orig_h)
            width, height = x2 - x1, y2 - y1
            
            face_landmarks = []
            if self.landmark_points > 0:
                for j in range(self.landmark_points):
                    lx = landmarks[0, i, 2*j] * orig_w
                    ly = landmarks[0, i, 2*j+1] * orig_h
                    face_landmarks.append((int(lx), int(ly)))
            
            faces.append({
                "box": (x1, y1, width, height),
                "confidence": confidence,
                "landmarks": face_landmarks
            })
        
        return sorted(faces, key=lambda x: x['confidence'], reverse=True)