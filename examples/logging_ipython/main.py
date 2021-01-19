import logging
import logging.config

from . import auxillary as aux
from . import config


def init_logging():
    """
    Make this a function so we can all it in main and in ipython.

    Ipython usage
    -------------
    >>> import examples.logging_ipython.main as elm
    >>> import examples.logging_ipython.auxillary as aux
    >>> elm.init_logging()
    >>> res = aux.my_add(1, 2)  # Should get a log message here.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Need this in order to get log info from other modules
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler"
            },
            "file": {
                "level": "INFO",
                "class": "logging.FileHandler",
                "filename": config.log_path
            }

        },
        "loggers": {
            "": {"handlers": ["console", "file"], "level": "INFO"}  # This is the root logger
        }
    }
    logging.config.dictConfig(logging_config)


logger = logging.getLogger(__name__)


def main():
    """
    Example where we should see log info from this function AND from my_add.
    """
    logger.info("In main")
    aux.my_add(x=1, y=2)


if __name__ == "__main__":
    init_logging()
    main()
