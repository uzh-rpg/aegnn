import torch
import torch_geometric
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as pl_metrics

from torch.nn.functional import softmax
from typing import Tuple
from .networks import by_name as model_by_name


class RecognitionModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                 dim: int = 3, learning_rate: float = 5e-3, **model_kwargs):
        super(RecognitionModel, self).__init__()
        self.optimizer_kwargs = dict(lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_outputs = num_classes
        self.dim = dim

        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        self.model = model_by_name(network)(dataset, model_input_shape, num_outputs=num_classes, **model_kwargs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.pos = data.pos[:, :self.dim]
        data.edge_attr = data.edge_attr[:, :self.dim]
        return self.model.forward(data)

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
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
        k = min(3, self.num_outputs - 1)
        self.log(f"Val/Accuracy_Top{k}", pl_metrics.accuracy(preds=predictions, target=batch.y, top_k=k))
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=5e-3, **self.optimizer_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRPolicy())
        return [optimizer], [lr_scheduler]


class LRPolicy(object):
    def __call__(self, epoch: int):
        if epoch <= 5:
            return 0.5 * (1 + epoch / 5)
        else:
            return 0.98 ** (epoch - 5)
