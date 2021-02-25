import numpy as np
import pytorch_lightning as pl
import wandb

from typing import Any, List


class ConfusionMatrix(pl.callbacks.base.Callback):
    """"

    Args:
        classes: list of class ids in the dataset, ordered in the same way as the groundtruth labels.
    """

    def __init__(self, classes: List[str]):
        self.classes = classes
        self.__y_hat = np.array([])
        self.__y_true = np.array([])

    def on_validation_batch_end(self, trainer, model: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Assuming the validation function returns the model predictions on the validation batch, in order
        to safe the computational cost of re-computing these predictions."""
        predictions_np = outputs.cpu().numpy()
        y_hat = np.argmax(predictions_np, axis=-1)
        self.__y_hat = np.concatenate([self.__y_hat, y_hat])
        y_np = batch.y.cpu().numpy()
        self.__y_true = np.concatenate([self.__y_true, y_np])

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if len(self.classes) < 10:
            cm = wandb.plot.confusion_matrix(y_true=self.__y_true, preds=self.__y_hat, class_names=self.classes,
                                             title=f"Confusion Matrix @ epoch {trainer.current_epoch}")
            model.logger.experiment.log({"conf_mat": cm}, commit=False)

        # Reset logs for next validation round.
        self.__y_hat = np.array([])
        self.__y_true = np.array([])
