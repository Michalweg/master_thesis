import logging
import os
from datetime import datetime
from pathlib import Path

ADD_FILE_HANDLER = True


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(add_file_handler=ADD_FILE_HANDLER):
    # Generate a filename with the current date and time (precision to minutes)
    log_filename = datetime.now().strftime("%d_%m_%Y_%H-%M.log")
    log_filename = Path("../logs" + f"/{log_filename}")

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set logging level

    # Create handlers
    console_handler = logging.StreamHandler()  # Console handler (logs to console)

    # Create a formatter and set it for both handlers
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    formatter = CustomFormatter()
    console_handler.setFormatter(CustomFormatter())

    # Add handlers to the logger
    logger.addHandler(console_handler)
    if add_file_handler:
        logging.basicConfig(
            filename=str(log_filename),
            filemode="a",  # Use 'w' to overwrite each time, 'a' to append
            format=formatter.format,
            level=logging.DEBUG,  # Set log level to DEBUG to capture all messages
        )
        file_handler = logging.FileHandler(
            log_filename, encoding="utf-8"
        )  # File handler (logs to file with custom filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger(ADD_FILE_HANDLER)
