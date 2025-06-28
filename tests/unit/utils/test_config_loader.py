import os
import pytest
from unittest.mock import patch, mock_open
from utils.config_loader import ConfigLoader

@pytest.fixture
def config_loader(tmp_path):
    # Create sample config files
    (tmp_path / "test_config.yaml").write_text("""
    key: "value"
    nested:
        num: 42
        env: "${TEST_ENV_VAR}"
    """)
    os.environ["TEST_ENV_VAR"] = "resolved_value"
    return ConfigLoader(config_dir=str(tmp_path))

def test_load_config(config_loader):
    config = config_loader.get_config("test_config")
    assert config["key"] == "value"
    assert config["nested"]["num"] == 42
    assert config["nested"]["env"] == "resolved_value"

def test_config_caching(config_loader):
    config1 = config_loader.get_config("test_config")
    config2 = config_loader.get_config("test_config")
    assert config1 is config2  # Should be same object from cache

def test_reload_config(config_loader):
    config1 = config_loader.get_config("test_config")
    
    # Mock file change
    with patch("builtins.open", mock_open(read_data="""
    key: "new_value"
    """)):
        config2 = config_loader.reload_config("test_config")
    
    assert config2["key"] == "new_value"
    assert config_loader.get_config("test_config")["key"] == "new_value"

def test_missing_config(config_loader):
    with pytest.raises(FileNotFoundError):
        config_loader.get_config("missing_config")

def test_complex_env_resolution(config_loader, tmp_path):
    (tmp_path / "env_config.yaml").write_text("""
    path1: "${HOME}/data"
    path2: "${DATA_DIR:-/default}/subdir"
    """)
    os.environ["HOME"] = "/user/home"
    
    config = config_loader.get_config("env_config")
    assert config["path1"] == "/user/home/data"
    assert config["path2"] == "/default/subdir"