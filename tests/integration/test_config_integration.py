import os
import pytest
from utils.config_loader import ConfigLoader

def test_config_loading_integration():
    config_loader = ConfigLoader()
    config = config_loader.get_config("pipeline")
    
    os.environ["DATA_DIR"] = "/test/path"
    config = config_loader.reload_config("pipeline")
    
    log_path = config["pipeline"]["logging"]["session_log_path"]
    resolved_path = log_path.replace("${DATA_DIR}", "/test/path")
    
    assert "/test/path" in resolved_path