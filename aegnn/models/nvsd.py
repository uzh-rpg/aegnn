import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import dropout, elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool_x, voxel_grid

from .base import DetectionModel
from .nvs import NVS


class NVSD(DetectionModel):

    def __init__(self, num_classes: int, num_bounding_boxes: int = 1, **kwargs):
        super().__init__(learning_rate=0.001, num_classes=num_classes, num_bounding_boxes=num_bounding_boxes)

        self.conv1 = SplineConv(1, out_channels=64, dim=2, kernel_size=5)
        self.norm1 = BatchNorm(in_channels=64)
        self.conv2 = SplineConv(64, out_channels=128, dim=2, kernel_size=5)
        self.norm2 = BatchNorm(in_channels=128)
        self.conv3 = SplineConv(128, out_channels=256, dim=2, kernel_size=5)
        self.norm3 = BatchNorm(in_channels=256)
        self.conv4 = SplineConv(256, out_channels=512, dim=2, kernel_size=5)
        self.norm4 = BatchNorm(in_channels=512)
        self.conv5 = SplineConv(512, out_channels=512, dim=2, kernel_size=5)
        self.norm5 = BatchNorm(in_channels=512)

        self.fc1 = Linear(16 * 512, out_features=1024)
        self.fc2 = Linear(1024, out_features=self.num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data = NVS.voxel_pooling(data, size=[4, 3])

        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        data = NVS.voxel_pooling(data, size=[16, 12])

        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data = NVS.voxel_pooling(data, size=[30, 23])

        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data = NVS.voxel_pooling(data, size=[40, 33])

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        cluster = voxel_grid(data.pos, data.batch, size=[60, 45])
        x, _ = max_pool_x(cluster, data.x, data.batch, size=16)

        x = x.view(-1, 16 * 512)
        x = elu(self.fc1(x))
        x = dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    #####################################################################################
    # Optimization ######################################################################
    #####################################################################################
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 110], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
