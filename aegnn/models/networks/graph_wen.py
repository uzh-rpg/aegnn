import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import dropout, elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX


class GraphWen(torch.nn.Module):

    def __init__(self, dataset: str, input_shape: torch.Tensor, num_outputs: int, **kwargs):
        super(GraphWen, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"

        self.block1 = ConvBlock(1, 64, voxel_size=4)
        self.block2 = ConvBlock(64, 128, voxel_size=8)
        self.block3 = ConvBlock(128, 256, voxel_size=16)
        self.block4 = ConvBlock(256, 512, voxel_size=32)

        self.pool_final = MaxPoolingX(input_shape[:2] // 2, size=4)
        self.linear1 = Linear(512 * 4, out_features=512,)
        self.linear2 = Linear(512, out_features=num_outputs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data = self.block1(data)
        data = self.block2(data)
        data = self.block3(data)
        data = self.block4(data)

        x = self.pool_final(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.linear1.in_features)
        x = self.linear1(x)
        x = dropout(x, p=0.3, training=self.training)
        return self.linear2(x)


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, voxel_size: int, transform=Cartesian(norm=True, cat=False)):
        super(ConvBlock, self).__init__()
        self.conv = SplineConv(in_channels, out_channels, dim=3, kernel_size=4)
        self.norm = BatchNorm(out_channels)
        self.pool = MaxPooling([voxel_size] * 3, transform=transform)

    def forward(self, data: torch_geometric.data.Batch):
        data.x = elu(self.conv(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm(data.x)
        data = self.pool(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)
        return data
