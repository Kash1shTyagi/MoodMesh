import numpy as np
from collections import deque
from typing import Dict, List, Deque
import time

class EmotionTracker:
    def __init__(self, config: dict):
        self.window_size = config['buffer_size']
        self.decay_rate = config['decay_rate']
        self.transition_threshold = config['transition_threshold']
        self.engagement_threshold = config['engagement_threshold']
        self.history: Dict[str, Deque[float]] = {}
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.stable_count = 0
        self.last_update = time.time()
        
    def update(self, emotions: Dict[str, float]):
        current_time = time.time()
        time_diff = current_time - self.last_update
        self.last_update = current_time
        
        for emotion in self.history:
            decay_factor = self.decay_rate ** time_diff
            self.history[emotion] = deque(
                [score * decay_factor for score in self.history[emotion]],
                maxlen=self.window_size
            )
        
        for emotion, score in emotions.items():
            if emotion not in self.history:
                self.history[emotion] = deque(maxlen=self.window_size)
            self.history[emotion].append(score)
            
        dominant, confidence = self._get_dominant()
        
        if self.current_emotion == dominant:
            self.stable_count += 1
        else:
            self.stable_count = max(0, self.stable_count - 1)
            
        if (confidence > self.current_confidence + self.transition_threshold or 
            (dominant != self.current_emotion and self.stable_count >= 3)):
            self.current_emotion = dominant
            self.current_confidence = confidence
            self.stable_count = 0
            
    def _get_dominant(self) -> tuple:
        if not self.history:
            return ("neutral", 0.0)
            
        avg_scores = {}
        for emotion, scores in self.history.items():
            if scores:
                avg_scores[emotion] = np.mean(list(scores))
        
        dominant = max(avg_scores, key=avg_scores.get)
        return (dominant, avg_scores[dominant])
    
    def get_dominant(self) -> str:
        return self.current_emotion
    
    def get_scores(self) -> Dict[str, float]:
        return {
            e: np.mean(list(scores)) if scores else 0.0
            for e, scores in self.history.items()
        }
    
    def get_engagement(self) -> float:
        """Calculate engagement score (1 - neutral confidence)"""
        scores = self.get_scores()
        neutral_score = scores.get("neutral", 0.0)
        return 1.0 - neutral_score
    
    def is_engaged(self) -> bool:
        """Check if user is emotionally engaged"""
        return self.get_engagement() > self.engagement_threshold
    
    def get_emotional_intensity(self) -> float:
        """Get overall emotional intensity (max of non-neutral emotions)"""
        scores = self.get_scores()
        non_neutral = {e: s for e, s in scores.items() 
                       if e != "neutral" and s > 0}
        return max(non_neutral.values()) if non_neutral else 0.0