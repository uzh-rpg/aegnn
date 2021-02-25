import math
import random
import torch
import torch_geometric

from torch_geometric.nn.pool import radius_graph, voxel_grid
from torch_geometric.transforms import Cartesian

from .base import Transform


class NVST(Transform):
    """Transform from events to computational graph as defined in NVS paper. Based on raw event data (x, y, t, p)
    perform the following transformations:

    - select a random temporal window with length `dt`
    - define each remaining event as node
    - coarsen graph by applying maximal count clustering (see below)
    - add an edge if the (weighted) spatio-temporal euclidean from one node to another is <= `r` with
      `d_max` as maximal number of possible neighbors
    - use the relative cartesian coordinates as edge attribute

    ["Graph-Based Object Classification for Neuromorphic VisionSensing" (Bi, 2019)]"""

    def __init__(self, r: float = 3, d_max: int = 32, dt: float = 0.03, beta: float = 0.5e-5, n_max: int = 8,
                 seed: int = 12345):
        random.seed(seed)
        self.r = float(r)
        self.d_max = int(d_max)
        self.dt = float(dt)  # 30 ms section
        self.beta = beta
        self.n_max = n_max

        self.__edge_attr = Cartesian(norm=True, cat=False)

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        # Filter a random window of length `dt` from the sample. To do so, find the number of
        # windows with length `dt` in the data, sample one of them and filter the data accordingly.
        if data.x.size(1) > 2:
            t_min = data.x[:, -2].min()
            t_max = data.x[:, -2].max()
            t_section = random.randint(1, int((t_max - t_min) / self.dt) - 1)

            t_start = t_min + self.dt * t_section
            t_end = t_min + self.dt * (t_section + 1)
            idx_select = torch.logical_and(t_start <= data.x[:, -2],  data.x[:, -2] < t_end)
            data.x = data.x[idx_select, -1:]  # polarity only {-1, +1}
            data.pos = data.pos[idx_select, :3]  # (x, y, t)
            if hasattr(data, "batch"):
                data.batch = data.batch[idx_select]

            # Re-weight temporal vs. spatial dimensions to account for different resolutions.
            data.pos[:, 2] = (data.pos[:, 2] - t_start) * math.sqrt(self.beta)

        # Coarsen graph by creating clusters with each cluster having maximal `n_max` nodes. Then
        # select one node (max node) from each cluster and drop the other ones.
        # clusters = self.max_count_clustering(data.pos, k=self.n_max)
        # data = self.sample_one_from_cluster(clusters=clusters, data=data)
        pseudo_batch = torch.zeros(data.num_nodes, device=data.x.device)
        cluster = voxel_grid(data.pos[:, :2], batch=pseudo_batch, size=[5, 5])
        data = self.sample_one_from_cluster(cluster, data=data)

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
        return f"{name}[r={self.r}, d_max={self.d_max}, dt={self.dt}, beta={self.beta}, n_max={self.n_max}]"

    #####################################################################################
    # Modules ###########################################################################
    #####################################################################################
    @staticmethod
    def max_count_clustering(data: torch.Tensor, k: int = 10) -> torch.Tensor:
        n = data.size(0)

        clusters = torch.zeros(n, 1, device=data.device)
        index = torch.arange(0, n, device=data.device).view(n, 1)
        data = torch.cat([data, index, clusters], dim=1)

        def split(points: torch.Tensor) -> torch.Tensor:
            if points.size(0) <= k:
                return points

            distances = torch.cdist(points[:, :-2], points[:, :-2], p=2)
            a, b = torch.nonzero(torch.eq(distances, torch.max(distances)))[0]
            in_a = torch.le(distances[a, :], distances[b, :])
            if torch.all(in_a) or not torch.any(in_a):
                half_length = int(in_a.numel() / 2)
                in_a[:half_length] = ~in_a[:half_length]

            points[in_a, -1] = random.randint(0, 99999999)
            points[~in_a, -1] = random.randint(0, 99999999)
            return torch.cat([split(points[in_a, :]), split(points[~in_a, :])])

        data = split(data)
        _, idx_sorted = torch.sort(data[:, -2], descending=False)
        return data[idx_sorted, -1]

    @staticmethod
    def sample_one_from_cluster(clusters: torch.Tensor, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        clusters_unique = clusters.unique(sorted=False)
        index_select = torch.zeros(clusters_unique.numel(), device=clusters.device).long()

        for i, cluster_id in enumerate(clusters_unique):
            is_in_cluster = torch.eq(clusters, cluster_id)
            cluster_idx = torch.nonzero(is_in_cluster)
            index_select[i] = cluster_idx[0]

        data.pos = data.pos[index_select, :]
        data.x = data.x[index_select, :]
        if hasattr(data, "batch"):
            data.batch = data.batch[index_select]
        return data
