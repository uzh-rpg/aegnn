import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import GMMConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from aegnn.models.base import DetectionModel


class RNLD(DetectionModel):

    def __init__(self, num_classes: int, num_bounding_boxes: int = 1, **kwargs):
        super().__init__(learning_rate=1e-3, num_classes=num_classes, num_bounding_boxes=num_bounding_boxes)
        self.conv1 = GMMConv(1, out_channels=8, dim=2, kernel_size=2)
        self.norm1 = BatchNorm(in_channels=8)
        self.conv2 = GMMConv(8, out_channels=16, dim=2, kernel_size=2)
        self.norm2 = BatchNorm(in_channels=16)

        self.conv3 = GMMConv(16, out_channels=16, dim=2, kernel_size=2)
        self.norm3 = BatchNorm(in_channels=16)
        self.conv4 = GMMConv(16, out_channels=16, dim=2, kernel_size=2)
        self.norm4 = BatchNorm(in_channels=16)

        self.conv5 = GMMConv(16, out_channels=32, dim=2, kernel_size=2)
        self.norm5 = BatchNorm(in_channels=32)

        self.fc1 = Linear(32 * 16, out_features=self.num_outputs * 2)
        self.fc2 = Linear(self.num_outputs * 2, out_features=self.num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)

        cluster = voxel_grid(data.pos, data.batch, size=[60, 45])
        x, _ = max_pool_x(cluster, data.x, data.batch, size=16)

        x = x.view(-1, self.fc1.in_features)
        x = elu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)