import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import wandb

from typing import Any, List


class ConfusionMatrix(pl.callbacks.base.Callback):
    """"Log confusion matrix for validation predictions.

    Args:
        classes: list of class ids in the dataset, ordered in the same way as the groundtruth labels.
    """

    def __init__(self, classes: List[str]):
        self.classes = np.array(classes)
        self.__y_hat = np.array([])
        self.__y_true = np.array([])

    def on_validation_batch_end(self, trainer, model: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Assuming the validation function returns the model predictions on the validation batch, in order
        to safe the computational cost of re-computing these predictions."""
        predictions_np = outputs.cpu().numpy()
        y_hat = np.argmax(predictions_np, axis=-1)
        self.__y_hat = np.concatenate([self.__y_hat, y_hat]).astype(int)
        y_np = batch.y.cpu().numpy()
        self.__y_true = np.concatenate([self.__y_true, y_np]).astype(int)

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        cm = sklearn.metrics.confusion_matrix(y_true=self.__y_true, y_pred=self.__y_hat)
        np.fill_diagonal(cm, 0)  # set correct elements to zero

        data = []
        max_index = cm.flatten().argsort()[-20:][::-1]
        for idx in max_index:
            i, j = np.unravel_index(idx, cm.shape)
            data.append([self.classes[i], self.classes[j], cm[i, j]])
        table = wandb.Table(data=data, columns=["True ClassID", "Predicted ClassID", "Number of Samples"])
        model.logger.experiment.log({"confusions": table}, step=trainer.global_step, commit=False)

        # Reset logs for next validation round.
        self.__y_hat = np.array([])
        self.__y_true = np.array([])
