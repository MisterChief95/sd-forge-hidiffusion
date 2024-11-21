import logging

from enum import Enum

try:
    from modules.shared import opts

    is_debug = getattr(opts, "is_debug", False)

except ImportError:
    is_debug = False


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class CustomFormatter(logging.Formatter):
    """
    CustomFormatter is a subclass of logging.Formatter that provides custom formatting for log messages with different colors based on the log level.
    Taken from: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    Attributes:
        grey (str): ANSI escape code for grey color.
        yellow (str): ANSI escape code for yellow color.
        red (str): ANSI escape code for red color.
        bold_red (str): ANSI escape code for bold red color.
        cyan (str): ANSI escape code for cyan color.
        white (str): ANSI escape code for white color.
        reset (str): ANSI escape code to reset color.
        log_format (str): Template for log message format.
        FORMATS (dict): Dictionary mapping log levels to their respective formatted log message templates.
    Methods:
        format(record):
            Formats the specified log record as text. The format is determined by the log level of the record.
    """

    grey = "\x1b[38m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[5;31m"
    cyan = "\x1b[36m"
    white = "\x1b[97m"
    reset = "\x1b[0m"
    log_format = "{1}[%(name)s]{0} {2}%(message)s{0}"

    FORMATS = {
        logging.DEBUG: log_format.format(reset, cyan, reset),
        logging.INFO: log_format.format(reset, cyan, white),
        logging.WARNING: log_format.format(reset, cyan, yellow),
        logging.ERROR: log_format.format(reset, cyan, red),
        logging.CRITICAL: log_format.format(reset, cyan, bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(log_fmt).format(record)


logger = logging.getLogger("HiDiffusion")
logger.setLevel(logging.INFO if not is_debug else logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not is_debug else logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())

logger.addHandler(console_handler)
