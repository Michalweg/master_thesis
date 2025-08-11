import logging
import os
from datetime import datetime
from pathlib import Path

ADD_FILE_HANDLER = True

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    log_format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.log_format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(add_file_handler=ADD_FILE_HANDLER):
    log_filename = Path("logs") / f"{datetime.now().strftime('%d_%m_%Y_%H-%M')}.log"
    os.makedirs(log_filename.parent, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Master level (capture all, handlers decide what to show)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler (no DEBUG spam)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only INFO and above for console
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # File handler (optional DEBUG logs)
    if add_file_handler:
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Keep DEBUG in file
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger(ADD_FILE_HANDLER)

# Example usage
logger.info("This should appear in both console and log file.")