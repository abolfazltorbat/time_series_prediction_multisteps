import logging
import platform

def setup_logger(log_file_path):
    """Set up the logger for logging messages."""
    logger = logging.getLogger('TimeSeriesModel')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('\n%(asctime)s - %(levelname)s - %(message)s\n' + '-' * 90)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def log_system_info(logger):
    """Log system information."""
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Node: {platform.node()}")
    logger.info(f"System: {platform.system()}")
    logger.info(f"Version: {platform.version()}")

