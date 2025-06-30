import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from services.emotion.recognition import EmotionRecognizer

@pytest.fixture
def recognizer_config():
    return {
        "model_path": "data/models/affectnet_emotion.onnx",
        "labels": ["neutral", "happy", "sad", "surprise", "anger", "disgust", "fear", "contempt"],
        "input_size": [64, 64],
        "threshold": 0.2
    }

@pytest.fixture
def mock_session():
    session = MagicMock()
    # Mock model output
    session.run.return_value = [np.array([[0.1, 0.6, 0.05, 0.05, 0.1, 0.05, 0.05, 0.0]])]
    return session

def test_recognize_emotions(recognizer_config, mock_session):
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        recognizer = EmotionRecognizer(recognizer_config)
        face_img = np.zeros((100, 100, 3), dtype=np.uint8)
        emotions = recognizer.recognize(face_img)
        
        assert "happy" in emotions
        assert emotions["happy"] == max(emotions.values())

def test_below_threshold(recognizer_config, mock_session):
    recognizer_config["threshold"] = 0.5
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        recognizer = EmotionRecognizer(recognizer_config)
        face_img = np.zeros((100, 100, 3), dtype=np.uint8)
        emotions = recognizer.recognize(face_img)
        
        assert len(emotions) == 1
        assert "happy" in emotions

def test_preprocessing(recognizer_config, mock_session):
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        recognizer = EmotionRecognizer(recognizer_config)
        face_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        recognizer.recognize(face_img)