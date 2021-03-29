import logging.config as lc
import yaml

from . import config


def init_logging_fc():
    lc.fileConfig(
        fname=config.log_ini_config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": config.log_output_path}
    )


def init_logging_yc():
    with open(config.log_yaml_config_path, "r") as f:
        log_config = yaml.safe_load(f)
    log_config["handlers"]["file"]["filename"] = config.log_output_path
    lc.dictConfig(log_config)
