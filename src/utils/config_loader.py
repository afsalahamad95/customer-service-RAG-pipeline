"""Configuration loader with environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Load YAML configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Recursively substitute environment variables
    config = _substitute_env_vars(config)
    
    return config


def _substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variables in configuration.
    
    Supports format: ${VAR_NAME:default_value}
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Check if string contains environment variable reference
        if obj.startswith("${") and obj.endswith("}"):
            # Extract variable name and default value
            var_spec = obj[2:-1]
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
            else:
                var_name, default = var_spec, None
            
            # Get value from environment or use default
            value = os.getenv(var_name, default)
            
            # Try to convert to appropriate type
            if value is not None:
                # Boolean conversion
                if value.lower() in ("true", "false"):
                    return value.lower() == "true"
                # Integer conversion
                try:
                    return int(value)
                except ValueError:
                    pass
                # Float conversion
                try:
                    return float(value)
                except ValueError:
                    pass
            
            return value
    
    return obj


def get_nested_config(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., "database.host")
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    keys = path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
