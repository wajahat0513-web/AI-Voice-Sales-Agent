from loguru import logger
import sys
from datetime import datetime
import os

# Create logs folder if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
log_file_path = os.path.join(LOG_DIR, f"app_log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure Loguru
logger.remove()  # Remove default logger
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
logger.add(
    log_file_path,
    rotation="10 MB",     
    retention="7 days",     
    level="INFO",
    encoding="utf-8",
    enqueue=True
)

def get_logger(name: str = None):
    """Return a child logger for a specific module"""
    return logger.bind(module=name or "general")
