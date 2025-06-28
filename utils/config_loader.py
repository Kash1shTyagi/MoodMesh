import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging
import re

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Advanced YAML configuration loader with environment variable substitution"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}
        load_dotenv() 
        
    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively replace ${VAR} with environment variables in strings"""
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def _replace(value: Any) -> Any:
            if isinstance(value, str):
                def replace_match(match):
                    var_name = match.group(1)
                    if ':-' in var_name:
                        var_name, default = var_name.split(':-', 1)
                        return os.getenv(var_name, default)
                    return os.getenv(var_name, match.group(0))
                return pattern.sub(replace_match, value)
            elif isinstance(value, dict):
                return {k: _replace(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_replace(item) for item in value]
            return value
        
        return _replace(config)
    
    def _load_yaml(self, config_name: str) -> Dict[str, Any]:
        """Load YAML file with caching"""
        if config_name in self._cache:
            return self._cache[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file {config_name}.yaml not found")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                resolved_config = self._resolve_env_vars(config)
                self._cache[config_name] = resolved_config
                return resolved_config
        except Exception as e:
            logger.exception(f"Error loading config {config_name}: {e}")
            raise
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get configuration by name"""
        return self._load_yaml(config_name)
    
    def get_full_config(self) -> Dict[str, Dict[str, Any]]:
        """Load all configurations"""
        return {
            "pipeline": self.get_config("pipeline"),
            "emotion": self.get_config("emotion"),
            "asr": self.get_config("asr"),
            "llm": self.get_config("llm"),
            "tts": self.get_config("tts")
        }
    
    def reload_config(self, config_name: Optional[str] = None):
        """Reload configuration(s) from disk"""
        if config_name:
            if config_name in self._cache:
                del self._cache[config_name]
            return self.get_config(config_name)
        else:
            self._cache = {}
            return self.get_full_config()


config_loader = ConfigLoader()