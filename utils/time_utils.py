import time
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FPSCounter:
    """Advanced FPS counter with smoothing and analytics"""
    
    def __init__(self, window_size: int = 30):
        self.times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self.frame_count = 0
        
    def update(self) -> float:
        """Update the counter and return current FPS"""
        current_time = time.perf_counter()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        if elapsed > 0:
            self.times.append(elapsed)
            self.frame_count += 1
        
        return self.current_fps
    
    def reset(self):
        """Reset the counter"""
        self.times.clear()
        self.frame_count = 0
        self.last_time = time.perf_counter()
    
    @property
    def current_fps(self) -> float:
        """Get current FPS (smoothed)"""
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))
    
    @property
    def min_fps(self) -> float:
        """Get minimum FPS in window"""
        if not self.times:
            return 0.0
        return 1.0 / max(self.times)
    
    @property
    def max_fps(self) -> float:
        """Get maximum FPS in window"""
        if not self.times:
            return 0.0
        return 1.0 / min(self.times)
    
    @property
    def jitter(self) -> float:
        """Get frame time jitter (std dev)"""
        if len(self.times) < 2:
            return 0.0
        return np.std(list(self.times))


class Timer:
    """Precision timer with multiple laps"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the timer"""
        self.start_time = time.perf_counter()
        self.last_lap = self.start_time
        self.laps = []
    
    def lap(self, name: str = None) -> float:
        """Record a lap time"""
        current_time = time.perf_counter()
        elapsed = current_time - self.last_lap
        self.last_lap = current_time
        
        if name:
            self.laps.append((name, elapsed))
        
        return elapsed
    
    def total(self) -> float:
        """Get total elapsed time"""
        return time.perf_counter() - self.start_time
    
    def get_lap_times(self) -> list:
        """Get all recorded lap times"""
        return self.laps.copy()
    
    def get_lap(self, name: str) -> float:
        """Get specific lap time by name"""
        for lap_name, time in self.laps:
            if lap_name == name:
                return time
        return 0.0


class Synchronizer:
    """Frame synchronizer for consistent processing rates"""
    
    def __init__(self, target_fps: float):
        self.target_interval = 1.0 / target_fps
        self.last_time = time.perf_counter()
        self.next_time = self.last_time + self.target_interval
    
    def wait_next(self):
        """Wait until next frame time"""
        current_time = time.perf_counter()
        sleep_time = self.next_time - current_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        self.last_time = self.next_time
        self.next_time += self.target_interval
        return self.last_time