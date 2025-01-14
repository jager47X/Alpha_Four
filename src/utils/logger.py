import logging
import os
import os

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

def setup_logger(log_file, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler for logging to a file
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler for logging to the console
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger


    return logger
def get_next_index(path):
    """
    Reads the directories in 'path' and returns the next available index as an integer.
    If no directories exist, returns 0.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return 0
    existing_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.isdigit()]
    existing_indices = [int(d) for d in existing_dirs]
    next_index = max(existing_indices) + 1 if existing_indices else 0
    return next_index
