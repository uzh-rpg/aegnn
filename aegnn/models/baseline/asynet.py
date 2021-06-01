import torch
import torch_geometric

from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear
from typing import Tuple

from ..base import DetectionModel


class AsyNet(DetectionModel):

    def __init__(self, num_classes: int, img_shape: Tuple[int, int], num_bounding_boxes: int = 1, **kwargs):
        super().__init__(num_classes, num_bounding_boxes=num_bounding_boxes, img_shape=img_shape, learning_rate=1e-3)

        self.kernel_size = 3
        self.conv_layers = Sequential(
            self.conv_block(in_channels=1, out_channels=16),
            self.conv_block(in_channels=16, out_channels=32),
            self.conv_block(in_channels=32, out_channels=64),
            self.conv_block(in_channels=64, out_channels=128),
            self.conv_block(in_channels=128, out_channels=256, max_pool=False),
            Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=2, bias=False),
            BatchNorm2d(512),
            ReLU(),
        )

        self.linear_input_features = int(self.input_shape[0] / 60) * int(self.input_shape[1] / 30) * 512
        self.linear_1 = Linear(self.linear_input_features, 1024)
        self.linear_2 = Linear(1024, self.num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        x = self.histogram(data.pos[:, :2], batch=data.batch, size=1, img_shape=self.input_shape)
        x = torch.unsqueeze(x, dim=1)
        x = x.clamp(min=0, max=20)  # maximal 20 events per bin

        x = self.conv_layers(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)

        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def conv_block(self, in_channels: int, out_channels: int, max_pool: bool = True):
        if max_pool:
            return Sequential(
                Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                BatchNorm2d(out_channels),
                ReLU(),
                Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                BatchNorm2d(out_channels),
                ReLU(),
                MaxPool2d(kernel_size=self.kernel_size, stride=2)
            )
        else:
            return Sequential(
                Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                BatchNorm2d(out_channels),
                ReLU(),
                Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                BatchNorm2d(out_channels),
                ReLU()
            )

    @staticmethod
    def histogram(x: torch.Tensor, batch: torch.LongTensor, img_shape: Tuple[int, int], size: float = 1):
        num_batch = int(batch.max().item())
        img_width_out = int(img_shape[0] / size)
        img_height_out = int(img_shape[1] / size)

        buckets_x = torch.arange(start=0, end=img_shape[0] - 1, step=size, device=x.device)
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
