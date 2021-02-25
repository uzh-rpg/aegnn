import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import dropout, elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool, max_pool_x, voxel_grid
from torch_geometric.transforms import Cartesian
from typing import List

from .base import MultiClassificationModel


class NVS(MultiClassificationModel):

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(learning_rate=0.001, num_classes=num_classes)
        self.conv1 = SplineConv(1, out_channels=64, dim=2, kernel_size=5)
        self.norm1 = BatchNorm(in_channels=64)
        self.conv2 = SplineConv(64, out_channels=128, dim=2, kernel_size=5)
        self.norm2 = BatchNorm(in_channels=128)
        self.conv3 = SplineConv(128, out_channels=256, dim=2, kernel_size=5)
        self.norm3 = BatchNorm(in_channels=256)
        self.conv4 = SplineConv(256, out_channels=512, dim=2, kernel_size=5)
        self.norm4 = BatchNorm(in_channels=512)
        self.fc1 = Linear(16 * 512, out_features=1024)
        self.fc2 = Linear(1024, out_features=num_classes)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data = self.voxel_pooling(data, size=[4, 3])

        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        data = self.voxel_pooling(data, size=[16, 12])

        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data = self.voxel_pooling(data, size=[30, 23])

        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        cluster = voxel_grid(data.pos, data.batch, size=[60, 45])
        x, _ = max_pool_x(cluster, data.x, data.batch, size=16)

        x = x.view(-1, 16 * 512)
        x = elu(self.fc1(x))
        x = dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

    #####################################################################################
    # Optimization ######################################################################
    #####################################################################################
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5, **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Val/Loss"}

    #####################################################################################
    # Modules ###########################################################################
    #####################################################################################
    @staticmethod
    def voxel_pooling(data: torch_geometric.data.Batch, size: List[float]) -> torch_geometric.data.Batch:
        """Max Pooling based on uniform-voxel-based clusters.

        The graph nodes are clustered using a uniformly-sized voxel grid, based on their positions (`pos`-attribute).
        Afterwards, max pooling is applied to each cluster, i.e. taking the maximum features out of each cluster.
        ["Graph-Based Object Classification for Neuromorphic VisionSensing" (Bi, 2019)]

        :param data: input graph data batch.
        :param size: size of voxel grid (same over all dimensions).
        """
        cluster = voxel_grid(data.pos[:, :2], batch=data.batch, size=size)
        return max_pool(cluster, data=data, transform=Cartesian(cat=False))  # transform for new edge attributes
