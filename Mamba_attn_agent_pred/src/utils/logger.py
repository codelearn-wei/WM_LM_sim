import os
import logging
from datetime import datetime

def setup_logger(name=None, log_file=None):
    """
    Setup logger for training.
    
    Args:
        name (str, optional): Name of the logger
        log_file (str, optional): Path to the log file. If None, a default path will be used.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file) if log_file else 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # If no log file specified, create one with timestamp
    if log_file is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'train_{current_time}.log')
    
    # Create logger
    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 