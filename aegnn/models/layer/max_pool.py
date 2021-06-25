import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid
from torch_geometric.transforms import Cartesian
from typing import List, Optional, Tuple, Union


class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int]):
        super(MaxPooling, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        cluster = voxel_grid(pos[:, :2], batch=batch, size=self.size)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=Cartesian(cat=False))  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
