import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from services.emotion.detection import FaceDetector

@pytest.fixture
def detector_config():
    return {
        "model_path": "data/models/buffalo_l/det_10g.onnx",
        "min_confidence": 0.7,
        "max_faces": 5,
        "input_size": [640, 640],
        "landmark_points": 5
    }

@pytest.fixture
def mock_session():
    session = MagicMock()
    # Mock InsightFace output format with scalar confidence scores
    session.run.return_value = [
        np.array([[[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]]]),  # bboxes
        np.zeros((1, 3, 10)),  # landmarks
        np.array([[0.9, 0.8, 0.6]])  # scores (scalar per face)
    ]
    return session

def test_detector_initialization(detector_config):
    with patch("onnxruntime.InferenceSession") as mock_session:
        detector = FaceDetector(detector_config)
        assert detector.min_confidence == 0.7
        assert detector.max_faces == 5
        mock_session.assert_called_once()

def test_detect_faces(detector_config, mock_session):
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        detector = FaceDetector(detector_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert isinstance(faces, list)
        assert len(faces) == 2  # 2 faces above confidence threshold

def test_no_faces_detected(detector_config, mock_session):
    mock_session.run.return_value[2] = np.array([[0.6, 0.5, 0.4]])
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        detector = FaceDetector(detector_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert len(faces) == 0

def test_landmark_extraction(detector_config, mock_session):
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        detector = FaceDetector(detector_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert len(faces) > 0
        assert len(faces[0]['landmarks']) == 5