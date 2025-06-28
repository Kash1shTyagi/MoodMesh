import time
import pytest
import numpy as np
from utils.time_utils import FPSCounter, Timer, Synchronizer

@pytest.fixture
def fps_counter():
    return FPSCounter(window_size=5)

@pytest.fixture
def timer():
    return Timer()

@pytest.fixture
def synchronizer():
    return Synchronizer(target_fps=10)  # 100ms per frame

def test_fps_counter_basic(fps_counter):
    for _ in range(5):
        time.sleep(0.05)
        fps_counter.update()
    
    assert 18 <= fps_counter.current_fps <= 22
    assert fps_counter.frame_count == 5

def test_fps_counter_jitter(fps_counter):
    intervals = [0.1, 0.05, 0.15, 0.02, 0.08]
    for interval in intervals:
        time.sleep(interval)
        fps_counter.update()
    
    assert fps_counter.jitter > 0.03  # Should have measurable jitter

def test_timer_basic(timer):
    time.sleep(0.1)
    lap1 = timer.lap("first")
    assert 0.09 <= lap1 <= 0.11
    
    time.sleep(0.2)
    lap2 = timer.lap("second")
    assert 0.19 <= lap2 <= 0.21
    
    assert timer.get_lap("first") == lap1
    assert timer.total() >= 0.3

def test_synchronizer(synchronizer):
    start = time.perf_counter()
    synchronizer.wait_next()  # First call immediate
    
    frame_times = []
    for _ in range(5):
        synchronizer.wait_next()
        frame_times.append(time.perf_counter())
    
    intervals = np.diff(frame_times)
    assert all(0.09 < interval < 0.11 for interval in intervals)