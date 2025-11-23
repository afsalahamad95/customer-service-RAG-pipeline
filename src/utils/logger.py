"""Centralized logging configuration using Loguru."""

import sys
from pathlib import Path
from loguru import logger

from src.utils.config_loader import load_config


def setup_logger():
    """Configure and return the application logger."""
    config = load_config()
    log_config = config.get("logging", {})
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    if "console" in log_config.get("output", ["console"]):
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_config.get("level", "INFO"),
            colorize=True,
        )
    
    # File handler
    if "file" in log_config.get("output", []):
        log_file_config = log_config.get("file", {})
        log_path = Path(log_file_config.get("path", "logs/rag_pipeline.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_config.get("level", "INFO"),
            rotation=log_file_config.get("rotation", "100 MB"),
            retention=log_file_config.get("retention", "30 days"),
            compression="zip",
            serialize=log_config.get("format") == "json",
        )
    
    return logger


# Global logger instance
log = setup_logger()


def get_logger(name: str):
    """Get a logger instance with a specific name."""
    return logger.bind(name=name)
