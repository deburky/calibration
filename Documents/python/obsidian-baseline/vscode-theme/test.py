from loguru import logger
from rich.logging import RichHandler
import sys

def setup_logger(level: str = "INFO", var=50):
    """Set up a logger with RichHandler for better formatting and color support."""
    logger.remove()  # Ensure no duplicated logs
    logger.add(sys.stdout, format="{message}")
    logger.configure(
        handlers=[{"sink": RichHandler(), "format": "{message}", "level": level}]
    )
    return logger

setup_logger()

