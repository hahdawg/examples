import logging

from . import module1
from . import module2
from . import util

logger = logging.getLogger(__name__)

util.init_logging()


def main():
    x = module1.fn()
    y = module2.fn()
    z = x + y
    logger.info("z = %s", z)
    return z


if __name__ == "__main__":
    main()
