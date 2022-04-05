import argparse
import logging
import os
import time

from pytorch_lightning.loggers import LightningLoggerBase
from typing import Dict, Optional, Any


class LoggingLogger(LightningLoggerBase):

    def __init__(self, save_dir: Optional[str], name: Optional[str] = "default", version: Optional[str] = None,
                 log_level: int = logging.INFO, sub_dir: Optional[str] = None, **kwargs):
        super(LoggingLogger, self).__init__()
        self._name = name
        self._version = version
        self._log_level = log_level
        self._sub_dir = sub_dir

        # If `save_dir` is set, setup output logging file in the log dir, following the directory naming convention
        # (base-directory)/(name)/version/... Otherwise setup logging in terminal.
        logging_kwargs = dict(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d %(levelname)s] %(message)s',
                              datefmt='%H:%M:%S')
        if save_dir is not None:
            log_dir = os.path.join(save_dir, name)
            if version is not None:
                log_dir = os.path.join(log_dir, version)
            if sub_dir is not None:
                log_dir = os.path.join(log_dir, sub_dir)
            os.makedirs(log_dir, exist_ok=True)

            filename = time.strftime("%Y%m%d-%H%M%S")
            logging.basicConfig(filename=os.path.join(log_dir, f"{filename}.log"), filemode='w+', **logging_kwargs)
        else:
            logging.basicConfig(**logging_kwargs)

    @property
    def experiment(self) -> Any:
        return None

    @property
    def sub_dir(self) -> Optional[str]:
        return self._sub_dir

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            msg = f"Logging {key} = {value}"
            if step is not None:
                msg += f" @ step {step}"
            self.__log_message(msg)

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        if len(args) > 0:
            logging.warning("Unnamed hyper-parameters cannot be logged")

        if type(params) == argparse.Namespace:
            params_dict = vars(params)
        elif type(params) == dict:
            params_dict = params
        else:
            logging.warning(f"Skipping params due to unknown type {type(params)}")
            params_dict = dict()

        kwargs.update(params_dict)  # adding params to kwargs dictionary
        for key, value in kwargs.items():
            msg = f"Hyper-Parameter {key} = {value}"
            self.__log_message(msg)

    def __log_message(self, msg: Any):
        logging.log(self._log_level, msg)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def save_dir(self) -> Optional[str]:
        log_file = getattr(logging.getLogger().handlers[-1], "baseFilename", None)
        if log_file is None:
            return None
        return os.path.dirname(log_file)
