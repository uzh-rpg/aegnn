import torch

from torch_geometric.nn.conv import SplineConv
from torch_geometric.transforms import Cartesian


from .base import AsyConvModule


class AsySplineConv(SplineConv, AsyConvModule):

    def __init__(self, in_channels: int, out_channels: int, dim: int, kernel_size: int, r: float,
                 do_graph_updates: bool = True, is_initial: bool = False, **kwargs):
        SplineConv.__init__(self, in_channels, out_channels, dim=dim, kernel_size=kernel_size,
                            bias=False, root_weight=False, **kwargs)
        AsyConvModule.__init__(self, module=SplineConv, r=r, do_graph_updates=do_graph_updates,
                               is_initial=is_initial, edge_attributes=Cartesian(cat=False, max_value=10.0))

    def forward(self, x: torch.Tensor, pos: torch.Tensor, **kwargs):
        return AsyConvModule.forward(self, x, pos, **kwargs)
