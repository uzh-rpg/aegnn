import torch
from torch_geometric.nn.conv import GCNConv

from .base import AsyConvModule


class AsyGCNConv(GCNConv, AsyConvModule):

    def __init__(self, in_channels: int, out_channels: int, r: float,
                 do_graph_updates: bool = True, is_initial: bool = False, **kwargs):
        GCNConv.__init__(self, in_channels, out_channels=out_channels, normalize=False, **kwargs)
        AsyConvModule.__init__(self, GCNConv, r, do_graph_updates=do_graph_updates, is_initial=is_initial)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, **kwargs):
        return AsyConvModule.forward(self, x, pos, **kwargs)
