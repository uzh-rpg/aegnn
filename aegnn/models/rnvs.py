import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import dropout, elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool_x, voxel_grid

from .base import MultiClassificationModel
from .nvs import NVS


class RNVS(MultiClassificationModel):

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(learning_rate=0.001, num_classes=num_classes)
        self.conv1 = SplineConv(1, out_channels=64, dim=2, kernel_size=5)
        self.norm = BatchNorm(in_channels=64)

        self.res1 = self.ResidualBlock(64, out_channel=128, dim=2, k=5)
        self.res2 = self.ResidualBlock(128, out_channel=256, dim=2, k=5)
        self.res3 = self.ResidualBlock(256, out_channel=512, dim=2, k=5)

        self.fc1 = Linear(64 * 512, out_features=1024)
        self.fc2 = Linear(1024, out_features=num_classes)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = self.conv1(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        data.x = elu(self.norm(data.x))
        data = NVS.voxel_pooling(data, size=[4, 3])

        data = self.res1(data)
        data = NVS.voxel_pooling(data, size=[16, 12])
        data = self.res2(data)
        data = NVS.voxel_pooling(data, size=[30, 23])

        data = self.res3(data)
        cluster = voxel_grid(data.pos[:, :2], batch=data.batch, size=[60, 45])
        x, _ = max_pool_x(cluster, x=data.x, batch=data.batch, size=64)

        x = x.view(-1, 64 * 512)
        x = elu(self.fc1(x))
        x = dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)  # no softmax since cross-entropy loss expects raw logits
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
    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channel: int, out_channel: int, dim: int, k: int = 5):
            super().__init__()
            self.left_conv1 = SplineConv(in_channel, out_channel, dim=dim, kernel_size=k)
            self.left_bn1 = BatchNorm(out_channel)
            self.left_conv2 = SplineConv(out_channel, out_channel, dim=dim, kernel_size=k)
            self.left_bn2 = BatchNorm(out_channel)

            self.shortcut_conv = SplineConv(in_channel, out_channel, dim=dim, kernel_size=1)
            self.shortcut_bn = BatchNorm(out_channel)

        def forward(self, data: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
            x_sc = data.x.clone()
            x_sc = self.shortcut_conv(x_sc, data.edge_index, data.edge_attr)
            x_sc = self.shortcut_bn(x_sc)

            x_res = self.left_conv1(data.x, data.edge_index, data.edge_attr)
            x_res = elu(self.left_bn1(x_res))
            x_res = self.left_conv2(x_res, data.edge_index, data.edge_attr)
            x_res = self.left_bn2(x_res)

            data.x = elu(x_sc + x_res)
            return data
