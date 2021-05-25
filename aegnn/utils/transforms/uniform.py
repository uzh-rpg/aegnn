import math
import torch
import torch_geometric

from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian, FixedPoints

from .base import Transform


class Uniform(Transform):

    def __init__(self, r: float = 3, d_max: int = 32, beta: float = 0.5e-5, n_max: int = 50000):
        self.r = float(r)
        self.d_max = int(d_max)
        self.beta = beta
        self.n_max = n_max

        self.__sampler = FixedPoints(num=self.n_max, allow_duplicates=False, replace=False)
        self.__edge_attr = Cartesian(norm=True, cat=False)

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = (data.pos[:, 2] - data.pos[:, 2].min()) * math.sqrt(self.beta)

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.__sampler(data)

        # Generate a graph based on the euclidean spatial distance between events. Due to
        # the difference resolution in the spatial and temporal dimensions, we have to
        # weigh them differently when computing the euclidean distance between nodes.
        data.edge_index = radius_graph(data.pos, r=self.r, max_num_neighbors=self.d_max)

        # The edge attributes thereby encodes the absolute relative spatial coordinates between nodes.
        data.pos = data.pos[:, :2]  # (x,y) only
        data = self.__edge_attr(data)  # egde-attributes = {|x|, |y|} (normalized)
        data.edge_attr[torch.isnan(data.edge_attr)] = 0  # nan -> 0
        return data

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}[r={self.r}, d_max={self.d_max}, beta={self.beta}, n_max={self.n_max}]"
