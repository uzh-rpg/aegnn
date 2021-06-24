import math
import numpy as np
import torch
import torch_geometric

from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian

from aegnn.octree import Box, OctTree
from .base import Transform


class NVST(Transform):
    """Transform from events to computational graph as defined in NVS paper. Based on raw event data (x, y, t, p)
    perform the following transformations:

    - select a random temporal window with length `dt`
    - define each remaining event as node
    - coarsen graph by applying uniform voxel grid clustering (see below)
    - add an edge if the (weighted) spatio-temporal euclidean from one node to another is <= `r` with
      `d_max` as maximal number of possible neighbors
    - use the relative cartesian coordinates as edge attribute

    ["Graph-Based Object Classification for Neuromorphic VisionSensing" (Bi, 2019)]"""

    def __init__(self, r: float = 10, d_max: int = 128, beta: float = 0.5e-5, n_max: int = 8):
        self.r = float(r)
        self.d_max = int(d_max)
        self.beta = beta
        self.n_max = n_max

        self.__edge_attr = Cartesian(norm=True, cat=False)

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = (data.pos[:, 2] - data.pos[:, 2].min()) * math.sqrt(self.beta)

        # Coarsen graph by creating clusters with each cluster having maximal `n_max` nodes. Then
        # select one node (max node) from each cluster and drop the other ones.
        clusters = self.oct_tree_clustering(data.pos, k=self.n_max)
        data = self.sample_one_from_cluster(clusters=clusters, data=data)

        # Generate a graph based on the euclidean spatial distance between events. Due to
        # the difference resolution in the spatial and temporal dimensions, we have to
        # weigh them differently when computing the euclidean distance between nodes.
        data.edge_index = radius_graph(data.pos, r=self.r, max_num_neighbors=self.d_max)

        # The edge attributes thereby encodes the absolute relative spatial coordinates between nodes.
        # data.pos = data.pos[:, :2]  # (x,y) only
        data = self.__edge_attr(data)  # egde-attributes = {|x|, |y|} (normalized)
        data.edge_attr[torch.isnan(data.edge_attr)] = 0  # nan -> 0
        return data

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}[r={self.r}, d_max={self.d_max}, beta={self.beta}, n_max={self.n_max}]"

    #####################################################################################
    # Modules ###########################################################################
    #####################################################################################
    @staticmethod
    def oct_tree_clustering(x: torch.Tensor, k: int) -> np.ndarray:
        if x.size(1) != 3:
            raise ValueError("Input tensor must be 3-dimensional!")
        x_min, y_min, z_min = x.min(dim=0).values.cpu().numpy()
        width = float(x[:, 0].max()) - x_min + 1e-3
        height = float(x[:, 1].max()) - y_min + 1e-3
        depth = float(x[:, 2].max()) - z_min + 1e-6
        domain = Box(width / 2, height / 2, depth / 2, width=width, height=height, depth=depth)

        oct_tree = OctTree(domain, max_points=k)
        x_tuples = [(float(px) - x_min, float(py) - y_min, float(pz) - z_min) for px, py, pz in x.cpu().numpy()]
        for point in x_tuples:
            oct_tree.insert(point)

        clusters = np.zeros(x.size(0))
        for i, point in enumerate(x_tuples):
            clusters[i] = oct_tree.assign(point)
        return clusters

    @staticmethod
    def sample_one_from_cluster(clusters: np.ndarray, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        _, index_select = np.unique(clusters, return_index=True)  # first occurrence of every unique value

        data.pos = data.pos[index_select, :]
        data.x = data.x[index_select, :]
        if hasattr(data, "batch"):
            data.batch = data.batch[index_select]
        return data
