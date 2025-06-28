import os
import pytest
from utils.config_loader import config_loader

def test_config_loading_integration():
    # This test assumes you have actual config files in configs/
    config = config_loader.get_config("pipeline")
    assert "pipeline" in config
    assert "mode" in config["pipeline"]
    
    # Test environment variable substitution
    os.environ["DATA_DIR"] = "/test/path"
    config = config_loader.reload_config("pipeline")
    assert "/test/path" in config["pipeline"]["logging"]["session_log_path"]