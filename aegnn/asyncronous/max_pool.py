import math

import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid
from torch_geometric.nn.pool.pool import pool_edge
from torch_geometric.typing import Adj
from torch_scatter import scatter_max, scatter_sum
from typing import Callable, List

from .base import AsyModule
from ..utils.tensors import intersection


class AsyMaxPool(AsyModule):
    """Sparsely updating maximum voxel-grid pooling operation.

    Args:
        size: voxel size in each dimension.
        grid_size: grid size (grid starting at 0, spanning to `grid_size`), >= `size`.
        r: radius for computing edge radius graph around new nodes.
        transform: A function/transform that takes in the coarsened and pooled :obj:`torch_geometric.data.Data` object
                   and returns a transformed version. (default = None).
    """
    # TODO: advance to dimensions > 2d using ravel multi index
    def __init__(self, size: List[int], grid_size: List[int], r: float,
                 do_graph_updates: bool = True, transform: Callable[[Data, ], Data] = None):
        super().__init__(do_graph_updates=do_graph_updates, r=r)

        assert len(size) == len(grid_size), "Voxel and grid dimensions must be identical"
        assert all([size[i] <= grid_size[i] for i in range(len(size))]), "voxels must be smaller or equal to grid"

        self.__graph_coarse = None  # coarse output graph
        self.__voxel_node_index = None  # index of max. node in input data
        self.__voxel_pos_sum = None  # sum of positions per voxel
        self.__voxel_node_count = None  # count of nodes per voxel

        self.__size = size  # voxel size in N dimensions
        self.__grid_size = grid_size  # grid size in N dimensions
        self.__transform = transform

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None):
        out = super().forward(x, pos, edge_index=edge_index)
        if self.__transform is not None:
            out = self.__transform(out)
        return out

    #####################################################################################
    # Graph operations ##################################################################
    #####################################################################################
    def graph_initialization(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr=None):
        graph_out = self._sync_process(x, pos, edge_index=None)
        self._graph = Data(x=x, pos=pos)

        self._graph.vdx = self.__get_clusters(pos)  # cluster index
        _, index = torch.unique(self._graph.vdx, sorted=True, return_inverse=True)
        _, argmax = scatter_max(x, index=index, dim=0)

        self.__voxel_node_index = torch.flatten(argmax).long()
        self.__voxel_pos_sum = scatter_sum(pos, index=index, dim=0)
        self.__voxel_node_count = scatter_sum(torch.ones_like(self._graph.vdx, device=x.device), index=index)
        self.__graph_coarse = graph_out.clone()

        return graph_out

    #####################################################################################
    # Asynchronous updates ##############################################################
    #####################################################################################
    def _async_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                       ) -> Data:
        _, diff_idx = self._graph_changed_nodes(x)
        _, new_idx = self._graph_new_nodes(x)

        replaced_idx = intersection(diff_idx.long(), self.__voxel_node_index)
        graph_vdx = getattr(self._graph, "vdx")
        for idx in replaced_idx:
            nodes_cluster = torch.nonzero(torch.eq(graph_vdx, graph_vdx[idx]))[:, 0]
            diff_idx = torch.cat([diff_idx, nodes_cluster])

            coarse_idx = torch.eq(self.__voxel_node_index, idx)
            self.__graph_coarse.x[coarse_idx] = -999999
            self.__voxel_pos_sum[coarse_idx] -= pos[idx]
            self.__voxel_node_count[coarse_idx] -= 1

        update_idx = torch.cat([diff_idx, new_idx])
        x_update = x[update_idx, :]
        pos_update = pos[update_idx, :]

        # Max/Average pool the features x and positions respectively.
        vdx_update = self.__get_clusters(pos_update)
        x_scatter = torch.cat([x_update, self.__graph_coarse.x])
        node_index_scatter = torch.cat([update_idx, self.__voxel_node_index])
        pos_sum_scatter = torch.cat([pos_update, self.__voxel_pos_sum])
        node_count_scatter = torch.cat([torch.ones_like(update_idx, device=x.device), self.__voxel_node_count])
        clusters = torch.cat([vdx_update, getattr(self.__graph_coarse, "vdx")])

        clusters_unique, index = torch.unique(clusters, sorted=True, return_inverse=True)
        x_max, argmax = scatter_max(x_scatter, index=index, dim=0)

        voxel_pos_sum = scatter_sum(pos_sum_scatter, index=index, dim=0)
        voxel_node_count = scatter_sum(node_count_scatter, index=index)
        pos_mean = torch.div(voxel_pos_sum, voxel_node_count.view(-1, 1))  # index is sorted, so output is too

        # The coarsened graph is reconnected by dropping all edges inside a voxel and
        # unifying all edges between voxels.
        vdx = torch.cat([getattr(self._graph, "vdx"), vdx_update[diff_idx.numel():]])
        vdx[diff_idx] = vdx_update[:diff_idx.numel()]
        edges_coarse = None
        if edge_index is not None:
            edges_coarse, _ = pool_edge(cluster=vdx, edge_index=edge_index, edge_attr=None)

        # Create the coarsened graph and update the internal graph. While the coarsened graph can be
        # overwritten completely, as it has been re-computed, most elements of the un-coarsened graph
        # are unchanged, and therefore only have to be partly updated.
        graph_out = Data(x=x_max, pos=pos_mean, edge_index=edges_coarse)
        if self.do_graph_updates:
            self._graph = Data(x=x, pos=pos, edge_index=edge_index, vdx=vdx).clone()

            self.__voxel_node_index = torch.flatten(node_index_scatter[argmax]).long()
            self.__voxel_pos_sum = voxel_pos_sum
            self.__voxel_node_count = voxel_node_count
            self.__graph_coarse = Data(x=graph_out.x, pos=graph_out.pos, edge_index=edges_coarse, vdx=clusters_unique)
        return graph_out

    def __get_clusters(self, pos: torch.Tensor) -> torch.LongTensor:
        # Voxel grid implementation in Pytorch-Geometric
        # https://github.com/rusty1s/pytorch_cluster/blob/906497e66488901a7b67d24709f0551261c975b5/csrc/cpu/grid_cpu.cpp
        assert len(self.__size) == pos.size()[1] == 2, "currently only 2D supported"
        num_rows = self.__num_voxels(dim=1)
        clusters = (pos[:, 1] // self.__size[1]) * num_rows + (pos[:, 0] // self.__size[0])
        return clusters.long()

    def __num_voxels(self, dim: int = None) -> int:
        num_dims = len(self.__size)
        num_voxels = [int(self.__grid_size[i] / self.__size[i]) + 1 for i in range(num_dims)]
        if dim is not None:
            return num_voxels[dim]
        return math.prod(num_voxels)

    #####################################################################################
    # Synchronous updates ###############################################################
    #####################################################################################
    def _sync_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                      ) -> Data:
        batch = torch.zeros(x.size()[0], device=x.device)  # no batch processing
        if edge_index is None:
            edge_index = self._compute_edges(pos)

        num_dims = pos.size()[-1]
        grid_start = torch.zeros(num_dims, device=x.device)
        grid_end = self.__grid_size
        clusters = voxel_grid(pos, batch=batch, size=self.__size, start=grid_start, end=grid_end)

        graph = Data(x=x, pos=pos, edge_index=edge_index, batch=None)
        graph_coarse = max_pool(clusters, data=graph, transform=self.__transform)
        graph_coarse.vdx = torch.unique(clusters, sorted=True)
        return graph_coarse
