import logging

logger = logging.getLogger(__name__)


def my_add(x, y):
    logger.info("In my_add")
    logger.info("  x = %s", x)
    logger.info("  y = %s", y)
    return x + y
