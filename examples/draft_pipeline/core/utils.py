"""
Utility functions for the embedding pipeline
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"pipeline_{module_name}_{datetime.now().strftime('%Y%m%d')}.log")
        ]
    )
    return logging.getLogger(module_name)


def print_banner(title: str):
    """Print a formatted banner"""
    width = 60
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path