import torch
import torch.nn.functional as F
import torch_geometric

from torch.nn import BatchNorm2d, Conv2d, Linear
from typing import Tuple

from .base import MultiClassificationModel


class HistCNN(MultiClassificationModel):
    """Simple MNIST-CNN based on the event histogram"""

    def __init__(self, num_classes: int, img_shape: Tuple[int, int], **kwargs):
        super().__init__(learning_rate=0.001, num_classes=num_classes)
        self.__img_shape = img_shape
        self.norm = BatchNorm2d(num_features=1)

        self.cnn1 = Conv2d(1, out_channels=32, kernel_size=3)
        self.norm1 = BatchNorm2d(num_features=32)
        img_shape = (img_shape[0] - 2) // 8, (img_shape[1] - 2) // 8  # 8 => max-pooling
        self.cnn2 = Conv2d(32, out_channels=64, kernel_size=3)
        self.norm2 = BatchNorm2d(num_features=64)
        img_shape = (img_shape[0] - 2) // 8, (img_shape[1] - 2) // 8   # 8 => max-pooling

        self.fn1 = Linear(64 * img_shape[0] * img_shape[1], out_features=num_classes)
        self.fn2 = Linear(num_classes, out_features=num_classes)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        x = self.histogram(data.pos[:, :2], batch=data.batch, size=1, img_shape=self.__img_shape)

        x = x.clamp(min=0, max=20)  # maximal 20 events per bin
        batch_size, img_width, img_height = x.size()
        x = x.view(batch_size, 1, img_width, img_height)
        x = self.norm(x)

        x = F.relu(self.cnn1(x))
        x = self.norm1(x)
        x = F.max_pool2d(x, 8)

        x = F.relu(self.cnn2(x))
        x = self.norm2(x)
        x = F.max_pool2d(x, 8)

        x = x.view(batch_size, -1)
        x = F.relu(self.fn1(x))
        x = self.fn2(x)  # no softmax since cross-entropy loss expects raw logits
        return x

    #####################################################################################
    # Optimization ######################################################################
    #####################################################################################
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.1)
        return [optimizer], [scheduler]

    #####################################################################################
    # Modules ###########################################################################
    #####################################################################################
    @staticmethod
    def histogram(x: torch.Tensor, batch: torch.LongTensor, img_shape: Tuple[int, int], size: float = 1):
        num_batch = int(batch.max().item())
        img_width_out = int(img_shape[0] / size)
        img_height_out = int(img_shape[1] / size)

        buckets_x = torch.arange(start=0, end=img_shape[0] - 1 , step=size, device=x.device)
        buckets_y = torch.arange(start=0, end=img_shape[1] - 1, step=size, device=x.device)

        x_dis = torch.bucketize(x[:, 0].contiguous(), boundaries=buckets_x)
        y_dis = torch.bucketize(x[:, 1].contiguous(), boundaries=buckets_y)
        xy_dis = x_dis + img_width_out * y_dis

        img = torch.zeros(num_batch + 1, img_width_out * img_height_out, device=x.device)
        for bi in range(num_batch + 1):
            unique, counts = xy_dis[batch == bi].unique(sorted=False, return_counts=True)
            unique = unique.long()
            counts = counts.float()
            img[bi, unique] = counts

        return img.view(num_batch + 1, img_height_out, img_width_out)
