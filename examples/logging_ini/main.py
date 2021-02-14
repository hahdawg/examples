import logging
import logging.config

from . import config
from . import module1
from . import module2

logger = logging.getLogger(__name__)


def init_logging():
    logging.config.fileConfig(
        fname=config.log_config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": config.log_output_path}
    )


init_logging()


def main():
    x = module1.fn()
    y = module2.fn()
    z = x + y
    logger.info("z = %s", z)
    return z


if __name__ == "__main__":
    main()
