import logging.config as lc

from . import config


def init_logging():
    lc.fileConfig(
        fname=config.log_config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": config.log_output_path}
    )
