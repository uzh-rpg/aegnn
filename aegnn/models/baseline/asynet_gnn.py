import torch
import torch_geometric

from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GMMConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from torch.nn import ReLU, Linear
from typing import List, Tuple


from ..base import DetectionModel
from ..layer import MaxPooling


class AsyNetGNN(DetectionModel):

    def __init__(self, num_classes: int, img_shape: Tuple[int, int], num_bounding_boxes: int = 1, **kwargs):
        super().__init__(num_classes, num_bounding_boxes=num_bounding_boxes, img_shape=img_shape, learning_rate=1e-3)

        self.kernel_size = 3
        self.dim = 3

        self.conv_layers = Sequential('x, pos, batch, edge_index, edge_attr', [
            (self.conv_block(in_channels=1, out_channels=16, pool_size=[4, 3]),
                'x, pos, batch, edge_index, edge_attr -> x, pos, batch, edge_index, edge_attr'),
            (self.conv_block(in_channels=16, out_channels=32, pool_size=[4, 3]),
                'x, pos, batch, edge_index, edge_attr -> x, pos, batch, edge_index, edge_attr'),
            (self.conv_block(in_channels=32, out_channels=64, pool_size=[4, 3]),
                'x, pos, batch, edge_index, edge_attr -> x, pos, batch, edge_index, edge_attr'),
            (self.conv_block(in_channels=64, out_channels=128, pool_size=[4, 3]),
                'x, pos, batch, edge_index, edge_attr -> x, pos, batch, edge_index, edge_attr'),
            (self.conv_block(in_channels=128, out_channels=256, pool_size=None),
                'x, pos, batch, edge_index, edge_attr -> x'),
            (GMMConv(in_channels=256, out_channels=512, dim=self.dim, kernel_size=self.kernel_size, bias=False),
                'x, edge_index, edge_attr -> x'),
            (BatchNorm(in_channels=512), 'x -> x'),
            (ReLU(inplace=True), 'x -> x'),
            (lambda x, pos, batch: (x, pos, batch), 'x, pos, batch -> x, pos, batch')
        ])

        self.linear_input_features = 16 * 512
        self.linear_1 = Linear(self.linear_input_features, 1024)
        self.linear_2 = Linear(1024, self.num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        x, pos, batch = self.conv_layers(data.x, data.pos, data.batch, data.edge_index, data.edge_attr)

        grid_size = self.input_shape // 4
        cluster = voxel_grid(pos, batch, size=grid_size)
        x, _ = max_pool_x(cluster, x, batch, size=16)

        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)

    def conv_block(self, in_channels: int, out_channels: int, pool_size: List[int] = None):
        modules = [
            (GMMConv(in_channels, out_channels=out_channels, dim=self.dim, kernel_size=self.kernel_size, bias=False),
                'x, edge_index, edge_attr -> x'),
            (BatchNorm(in_channels=out_channels), 'x -> x'),
            (ReLU(inplace=True), 'x -> x'),
            (GMMConv(out_channels, out_channels=out_channels, dim=self.dim, kernel_size=self.kernel_size, bias=False),
                'x, edge_index, edge_attr -> x'),
            (BatchNorm(in_channels=out_channels), 'x -> x'),
            (ReLU(inplace=True), 'x -> x')
        ]

        if pool_size is not None:
            max_pool = (MaxPooling(size=pool_size), 'x, pos, batch, edge_index -> x, pos, batch, edge_index, edge_attr')
            modules.append(max_pool)
        return Sequential('x, pos, batch, edge_index, edge_attr', modules)
