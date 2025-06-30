import pytest
import time
import numpy as np
from services.emotion.tracker import EmotionTracker

@pytest.fixture
def tracker_config():
    return {
        "buffer_size": 5,
        "decay_rate": 0.95,
        "transition_threshold": 0.2,
        "engagement_threshold": 0.4
    }

def test_emotion_transition(tracker_config):
    tracker = EmotionTracker(tracker_config)
    
    tracker.update({"happy": 0.8, "neutral": 0.2})
    assert tracker.get_dominant() == "happy"
    
    tracker.update({"happy": 0.5, "neutral": 0.5})
    assert tracker.get_dominant() == "happy"  
    
    # Force transition
    for _ in range(4):
        tracker.update({"neutral": 0.7})
    assert tracker.get_dominant() == "neutral"

def test_engagement_detection(tracker_config):
    tracker = EmotionTracker(tracker_config)
    
    # Neutral state
    tracker.update({"neutral": 0.9, "happy": 0.1})
    assert not tracker.is_engaged()
    assert tracker.get_engagement() == pytest.approx(0.1, abs=0.01)
    
    # Engaged state (create a new tracker to avoid decay effects)
    tracker = EmotionTracker(tracker_config)
    tracker.update({"happy": 0.8, "surprise": 0.2})
    assert tracker.is_engaged()
    assert tracker.get_engagement() == 1.0 

def test_emotional_intensity(tracker_config):
    tracker = EmotionTracker(tracker_config)
    tracker.update({"anger": 0.9, "disgust": 0.8})
    assert tracker.get_emotional_intensity() == 0.9
    
    tracker = EmotionTracker(tracker_config)
    tracker.update({"neutral": 0.9})
    assert tracker.get_emotional_intensity() == 0.0

def test_time_based_decay(tracker_config):
    tracker = EmotionTracker(tracker_config)
    tracker.update({"surprised": 1.0})
    assert tracker.get_scores()["surprised"] == 1.0
    
    time.sleep(0.1)
    tracker.update({"surprised": 0.9})
    surprised_score = tracker.get_scores()["surprised"]
    assert 0.85 < surprised_score < 0.95  # Should have decayed

def test_emotional_intensity_without_neutral(tracker_config):
    tracker = EmotionTracker(tracker_config)
    tracker.update({"happy": 0.8, "excited": 0.7})
    assert tracker.get_emotional_intensity() == 0.8