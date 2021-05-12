import argparse
from typing import Callable, Union, Dict, Optional, Any

from pytorch_lightning.loggers.base import LightningLoggerBase


class PrintLogger(LightningLoggerBase):

    def __init__(self, filter_key: Callable = None, name: str = "logger"):
        super().__init__()
        self._name = name
        self._filter_key = filter_key

    @property
    def experiment(self) -> Any:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._filter_key is not None:
            metrics = {key: value for key, value in metrics.items() if self._filter_key(key)}
        if len(metrics) > 0:
            print(f"{metrics} @ step = {step}")

    def log_hyperparams(self, params: argparse.Namespace):
        print(params)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Union[int, str]:
        return "none"
