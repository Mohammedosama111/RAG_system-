"""
Configuration loader for the RAG system
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigLoader:
    """Load and manage system configuration"""
    
    _config_cache: Optional[Dict[str, Any]] = None
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file and environment variables
        """
        if cls._config_cache is not None:
            return cls._config_cache
        
        # Load environment variables
        load_dotenv()
        
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        config = {}
        
        # Load from YAML file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}")
        
        # Override with environment variables
        cls._override_with_env_vars(config)
        
        # Cache the config
        cls._config_cache = config
        return config
    
    @classmethod
    def _override_with_env_vars(cls, config: Dict[str, Any]) -> None:
        """Override configuration with environment variables"""
        
        # Gemini API key
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            if 'gemini' not in config:
                config['gemini'] = {}
            config['gemini']['api_key'] = gemini_api_key
        
        # Environment
        environment = os.getenv('ENVIRONMENT', 'development')
        config['environment'] = environment
        
        # Log level
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = log_level
    
    @classmethod
    def get(cls, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: ConfigLoader.get('gemini.api_key', 'default-key')
        """
        config = cls.load_config()
        keys = key_path.split('.')
        
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @classmethod
    def reload_config(cls) -> Dict[str, Any]:
        """Reload configuration (useful for testing)"""
        cls._config_cache = None
        return cls.load_config()