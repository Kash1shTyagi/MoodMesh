import numpy as np
import onnxruntime as ort
import cv2
from typing import Dict, Any

class EmotionRecognizer:
    def __init__(self, config: dict):
        self.config = config
        self.model = self._load_model(config['model_path'])
        self.labels = config['labels']
        self.input_size = tuple(config['input_size'])
        self.threshold = config['threshold']
        
    def _load_model(self, model_path: str) -> ort.InferenceSession:
        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        return ort.InferenceSession(
            model_path,
            providers=providers
        )
    
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(gray, self.input_size)
        
        img = img.astype(np.float32)
        img = img / 255.0
        img = img - 0.5
        img = img * 2.0
        
        img = np.expand_dims(img, axis=0)  
        return np.expand_dims(img, axis=0) 
    
    def recognize(self, face_img: np.ndarray) -> Dict[str, float]:
        input_data = self._preprocess_face(face_img)
        
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_data})[0][0]
        
        exp = np.exp(outputs - np.max(outputs))
        probs = exp / exp.sum()
        
        results = {}
        for i, label in enumerate(self.labels):
            if probs[i] >= self.threshold:
                results[label] = float(probs[i])
        
        if not results:
            max_idx = np.argmax(probs)
            results[self.labels[max_idx]] = float(probs[max_idx])
            
        return results