import time
import pytest
from unittest.mock import MagicMock, patch
from utils.system_monitor import SystemMonitor

@pytest.fixture
def monitor():
    return SystemMonitor(interval=0.1)

def test_monitor_start_stop(monitor):
    monitor.start()
    time.sleep(0.3)
    monitor.stop()
    
    assert len(monitor.data["cpu"]) >= 2
    assert len(monitor.data["memory"]) >= 2
    assert len(monitor.data["process"]) >= 2

@patch("psutil.cpu_percent")
@patch("psutil.virtual_memory")
@patch("GPUtil.getGPUs")
def test_monitor_data_collection(mock_gpus, mock_mem, mock_cpu, monitor):
    # Mock system responses
    mock_cpu.return_value = 30.0
    mock_mem.return_value.percent = 50.0
    mock_gpus.return_value = [MagicMock(load=0.5)]  # 50% GPU load
    
    monitor.start()
    time.sleep(0.15)
    monitor.stop()
    
    assert monitor.get_current("cpu") == 30.0
    assert monitor.get_current("memory") == 50.0
    assert monitor.get_current("gpu") == 50.0
    assert monitor.get_process_stats()["memory"] > 0

def test_alert_thresholds(monitor):
    monitor.alert_thresholds["cpu"] = 80
    monitor.set_threshold("memory", 85)
    
    # Mock high CPU usage
    with patch("psutil.cpu_percent", return_value=85.0), \
         patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.percent = 70.0
        monitor.start()
        time.sleep(0.15)
        monitor.stop()
    
    assert len(monitor.alerts) == 1
    assert monitor.alerts[0]["resource"] == "CPU"
    assert monitor.alerts[0]["value"] == 85.0

def test_process_monitoring(monitor):
    monitor.start()
    time.sleep(0.15)
    monitor.stop()
    
    process_stats = monitor.get_process_stats()
    assert "cpu" in process_stats
    assert "memory" in process_stats
    assert process_stats["memory"] > 0  # Should have some memory usage