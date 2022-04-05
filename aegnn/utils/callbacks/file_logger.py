import logging
import os
import inspect
import pytorch_lightning as pl

from typing import Any, List


class FileLogger(pl.callbacks.base.Callback):
    """Logging the code defining the given objects.
    For versioning of model and data processing usually a github commit is used. However, a commit
    uploads all of the files every time, while only tiny bits of code have changed, making it very
    redundant. This callback gives the opportunity to only upload specific code, e.g. only the
    code defining data processing and the model.
    Args:
        objects: class objects to be logged (code file).
    """

    def __init__(self, objects: List[Any]):
        self.objects = objects

    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        objects_flat_a = [[x] for x in self.objects if type(x) != list]
        objects_flat_b = [x for x in self.objects if type(x) == list]
        for obj in sum(objects_flat_a + objects_flat_b, []):
            self.__log_object_file(obj, logger=model.logger)

    @staticmethod
    def __log_object_file(obj, logger):
        experiments = logger.experiment
        if not type(experiments) == list:
            experiments = [experiments]
        logging.debug(f"Found {len(experiments)} for save file logs")

        for exp in experiments:
            if obj is not None and hasattr(exp, "save"):
                obj_file = inspect.getfile(obj.__class__)
                logging.debug(f"Logging file {obj_file}")
                exp.save(obj_file, base_path=os.path.dirname(obj_file))
