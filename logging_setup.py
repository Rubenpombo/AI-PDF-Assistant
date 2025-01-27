import logging
import os
from logging.handlers import RotatingFileHandler

def setup_log():
    """
    Sets up the logging configuration for the application.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a rotating file handler to manage log file size
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(current_dir, 'app.log')
    
    # Set up RotatingFileHandler with a max file size of 10 MB and keep 5 backups
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Check if the handler is already added to avoid duplicate logging
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger
