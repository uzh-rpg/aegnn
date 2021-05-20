import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from aegnn.models.base import DetectionModel


class CNLD(DetectionModel):

    def __init__(self, num_classes: int, num_bounding_boxes: int = 1, **kwargs):
        super().__init__(learning_rate=1e-3, num_classes=num_classes, num_bounding_boxes=num_bounding_boxes)
        self.conv = GCNConv(1, out_channels=8)
        self.norm = BatchNorm(in_channels=8)
        self.fc = Linear(16 * 8, out_features=self.num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.x = elu(self.conv(data.x, data.edge_index))
        data.x = self.norm(data.x)

        cluster = voxel_grid(data.pos, data.batch, size=[60, 45])
        x, _ = max_pool_x(cluster, data.x, data.batch, size=16)

        x = x.view(-1, 8 * 16)
        x = self.fc(x)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)
