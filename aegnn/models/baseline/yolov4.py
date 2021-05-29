import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import leaky_relu
from torch_geometric.data import Batch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool, max_pool_x, voxel_grid
from torch_geometric.transforms import Cartesian
from typing import Tuple

from ..base import DetectionModel


class YoloV4(DetectionModel):

    def __init__(self, num_classes: int, img_shape: Tuple[int, int], num_bounding_boxes: int = 1, **kwargs):
        super().__init__(num_classes, num_bounding_boxes=num_bounding_boxes, img_shape=img_shape, learning_rate=1e-3)
        self.conv1 = GCNConv(1, out_channels=8)
        self.conv2 = GCNConv(8, out_channels=16)
        self.conv3 = GCNConv(16, out_channels=32)
        self.norm3 = BatchNorm(in_channels=32)

        self.conv41 = GCNConv(32, out_channels=16)
        self.conv42 = GCNConv(16, out_channels=16)
        self.norm4 = BatchNorm(in_channels=32)
        self.conv43 = GCNConv(32, out_channels=32)

        self.conv5 = GCNConv(64, out_channels=64)
        # self.conv6 = GCNConv(64, out_channels=64)
        # self.norm6 = BatchNorm(in_channels=64)

        self.fc1 = Linear(64 * 16, out_features=self.num_outputs * 2)
        self.fc2 = Linear(self.num_outputs * 2, out_features=self.num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = leaky_relu(self.conv1(data.x, data.edge_index))
        data.x = leaky_relu(self.conv2(data.x, data.edge_index))
        data.x = leaky_relu(self.conv3(data.x, data.edge_index))
        data.x = self.norm3(data.x)

        x_sc_outer = data.x.clone()

        data.x = leaky_relu(self.conv41(data.x, data.edge_index))
        x_sc_inner = data.x.clone()
        data.x = leaky_relu(self.conv42(data.x, data.edge_index))
        data.x = torch.cat([x_sc_inner, data.x], dim=-1)
        data.x = self.norm4(data.x)
        data.x = leaky_relu(self.conv43(data.x, data.edge_index))

        data.x = torch.cat([x_sc_outer, data.x], dim=-1)
        data.x = leaky_relu(self.conv5(data.x, data.edge_index))

        # grid_size = self.input_shape // 2
        # cluster = voxel_grid(data.pos[:, :2], batch=data.batch, size=grid_size)
        # data: Batch = max_pool(cluster, data=data, transform=Cartesian(cat=False))
        #
        # x_sc_outer = data.x.clone()
        # data.x = leaky_relu(self.conv6(data.x, data.edge_index))
        # data.x = self.norm6(data.x)
        # data.x = data.x + x_sc_outer

        grid_size = self.input_shape // 4
        cluster = voxel_grid(data.pos, data.batch, size=grid_size)
        x, _ = max_pool_x(cluster, data.x, data.batch, size=16)  # 4 * 4 = 16

        x = x.view(-1, self.fc1.in_features)
        x = leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)
