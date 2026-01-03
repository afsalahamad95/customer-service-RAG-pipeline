"""Centralized logging configuration using Loguru."""

import os
import sys
from pathlib import Path

from loguru import logger


def setup_logger():
    """Configure and return the application logger."""
    # Get log level from environment or use INFO as default
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # Remove default handler
    logger.remove()

    # Console handler (always enabled)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler
    log_path = Path(os.getenv("LOG_FILE_PATH", "logs/rag_pipeline.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    return logger


# Global logger instance
log = setup_logger()


def get_logger(name: str):
    """Get a logger instance with a specific name."""
    return logger.bind(name=name)
