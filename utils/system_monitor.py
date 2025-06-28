import psutil
import time
import threading
from typing import Dict, Any, Optional
import logging
import os
import GPUtil

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Advanced system resource monitor with alerting"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.data = {
            "cpu": [],
            "memory": [],
            "gpu": [],
            "disk": [],
            "network": [],
            "process": []
        }
        self.alerts = []
        self.alert_thresholds = {
            "cpu": 90,
            "memory": 85,
            "gpu": 80,
            "disk": 90
        }
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
    
    def start(self):
        """Start monitoring thread"""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        logger.info("System monitor started")
    
    def stop(self):
        """Stop monitoring thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("System monitor stopped")
    
    def _monitor(self):
        """Monitoring loop"""
        while self._running:
            try:
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                self.data["cpu"].append(cpu_percent)
                
                # Memory monitoring
                mem = psutil.virtual_memory()
                self.data["memory"].append(mem.percent)
                
                # GPU monitoring
                gpu_percent = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = gpus[0].load * 100
                    self.data["gpu"].append(gpu_percent)
                except:
                    self.data["gpu"].append(0)
                
                # Disk monitoring
                disk = psutil.disk_usage('/')
                self.data["disk"].append(disk.percent)
                
                # Network monitoring
                net = psutil.net_io_counters()
                self.data["network"].append({
                    "sent": net.bytes_sent,
                    "recv": net.bytes_recv
                })
                
                # Process monitoring
                process_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
                self.data["process"].append({
                    "cpu": self.process.cpu_percent(),
                    "memory": process_mem
                })
                
                # Check thresholds
                self._check_alerts(cpu_percent, mem.percent, gpu_percent, disk.percent)
                
                # Prune data
                for key in self.data:
                    if len(self.data[key]) > 300:  # Keep 5 minutes at 1s intervals
                        self.data[key] = self.data[key][-300:]
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def _check_alerts(self, cpu: float, mem: float, gpu: float, disk: float):
        """Check resource usage against thresholds"""
        thresholds = self.alert_thresholds
        
        if cpu > thresholds["cpu"]:
            self._trigger_alert("CPU", cpu)
        
        if mem > thresholds["memory"]:
            self._trigger_alert("Memory", mem)
        
        if gpu > thresholds["gpu"]:
            self._trigger_alert("GPU", gpu)
        
        if disk > thresholds["disk"]:
            self._trigger_alert("Disk", disk)
    
    def _trigger_alert(self, resource: str, value: float):
        """Trigger an alert"""
        alert = {
            "resource": resource,
            "value": value,
            "threshold": self.alert_thresholds[resource.lower()],
            "timestamp": time.time()
        }
        
        # Only alert if not already alerted recently
        if not self.alerts or self.alerts[-1]["resource"] != resource:
            self.alerts.append(alert)
            logger.warning(f"Resource alert: {resource} usage {value}% exceeds threshold")
    
    def get_recent(self, metric: str, seconds: int = 10) -> list:
        """Get recent metric values"""
        count = min(len(self.data.get(metric, [])), seconds)
        return self.data.get(metric, [])[-count:]
    
    def get_current(self, metric: str) -> Any:
        """Get current metric value"""
        if not self.data.get(metric):
            return None
        return self.data[metric][-1] if self.data[metric] else None
    
    def get_process_stats(self) -> Dict[str, float]:
        """Get current process statistics"""
        if not self.data["process"]:
            return {}
        return self.data["process"][-1]
    
    def set_threshold(self, resource: str, value: float):
        """Set alert threshold for a resource"""
        resource = resource.lower()
        if resource in self.alert_thresholds:
            self.alert_thresholds[resource] = value
            return True
        return False


system_monitor = SystemMonitor()