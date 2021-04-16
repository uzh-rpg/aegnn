import torch
import torch_geometric
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as pl_metrics

from torch.nn.functional import softmax


class MultiClassificationModel(pl.LightningModule):

    def __init__(self, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.optimizer_kwargs = dict(lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> float:
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=batch.y)

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = pl_metrics.accuracy(preds=y_prediction, target=batch.y)
        self.logger.log_metrics({"Train/Loss": loss, "Train/Accuracy": accuracy}, step=self.trainer.global_step)
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)

        self.log("Val/Loss", self.criterion(outputs, target=batch.y))
        self.log("Val/Accuracy", pl_metrics.accuracy(preds=y_prediction, target=batch.y))
        k = min(3, self.num_classes - 1)
        self.log(f"Val/Accuracy_Top{k}", pl_metrics.accuracy(preds=predictions, target=batch.y, top_k=k))
        return predictions
