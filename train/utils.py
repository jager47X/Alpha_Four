import logging
import os
import torch
import torch.nn as nn
from models import DQN
from replay_buffer import DiskReplayBuffer


def safe_make_dir(directory):
    """
    Creates the specified directory if it doesn't exist.
    
    Args:
        directory (str): The path of the directory to create.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ensured: {directory}")
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")
        raise


def setup_logger(log_file, level=logging.INFO):
    """
    Sets up a logger that logs both to a file (log_file) and the console (stdout).
    Ensures the directory for log_file exists before creating the FileHandler.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicated logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure the directory for log_file exists
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            safe_make_dir(log_dir)

    # Create a file handler if log_file is provided
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    # Always add a stream handler (for console logs)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

    return logger


def get_next_index(path):
    """
    Reads the directories in 'path' and returns the next available index as an integer.
    If no directories exist, returns 0.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return 0
    existing_dirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.isdigit()
    ]
    existing_indices = [int(d) for d in existing_dirs]
    next_index = max(existing_indices) + 1 if existing_indices else 0
    return next_index
