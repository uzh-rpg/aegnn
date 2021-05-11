import torch

from torch_geometric.nn.conv import SplineConv
from torch_geometric.data import Data
from torch_geometric.transforms import Cartesian
from torch_geometric.typing import Adj, Size
from torch_geometric.utils import k_hop_subgraph, remove_self_loops

from .base import AsyModule


class AsySplineConv(SplineConv, AsyModule):
    # TODO: Graph updates
    # TODO: add bias and root weight

    def __init__(self, in_channels: int, out_channels: int, dim: int, kernel_size: int, r: float,
                 do_graph_updates: bool = True, is_initial: bool = False, **kwargs):
        SplineConv.__init__(self, in_channels, out_channels, dim=dim, kernel_size=kernel_size,
                            bias=False, root_weight=False, **kwargs)
        AsyModule.__init__(self, do_graph_updates=do_graph_updates, r=r)
        self.__is_initial = is_initial
        self.__edge_attr = Cartesian(cat=False, max_value=10.0)

    #####################################################################################
    # Graph operations ##################################################################
    #####################################################################################
    def graph_initialization(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr=None):
        if edge_attr is None:
            if edge_index is None:
                edge_index = self._compute_edges(pos)
            attr_data = Data(pos=pos, edge_index=edge_index)
            edge_attr = self.__edge_attr(attr_data).edge_attr

        y = SplineConv.forward(self, x, edge_index=edge_index, edge_attr=edge_attr)
        self._graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return self._graph.y

    def forward(self, x: torch.Tensor, pos: torch.Tensor, **kwargs):
        return AsyModule.forward(self, x, pos, **kwargs)

    #####################################################################################
    # Asynchronous updates ##############################################################
    #####################################################################################
    def _async_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                       ) -> torch.Tensor:
        if not self.__is_initial:
            x_new, idx_new = self._graph_new_nodes(x)
            pos_new = pos[idx_new, :]
            _, idx_diff = self._graph_changed_nodes(x)
            idx_diff, _, _, _ = k_hop_subgraph(idx_diff, num_hops=1, edge_index=self._graph.edge_index)
            x_all = x
            pos_all = pos
        else:
            num_prev_nodes = self._graph.num_nodes
            x_new, pos_new = x, pos
            idx_new = torch.arange(num_prev_nodes, num_prev_nodes + x.size()[0], device=x.device)
            idx_diff = torch.tensor([], device=x.device, dtype=torch.long)
            x_all = torch.cat([self._graph.x, x_new], dim=0)
            pos_all = torch.cat([self._graph.pos, pos_new], dim=0)

        connected_node_mask = (torch.cdist(pos_all, pos_new) <= self._radius)
        idx_update = torch.cat([torch.unique(torch.nonzero(connected_node_mask)[:, 0]), idx_diff])
        _, edges_connected, _, connected_edges_mask = k_hop_subgraph(idx_update, num_hops=1,
                                                                     edge_index=self._graph.edge_index,
                                                                     num_nodes=pos_all.size()[0])
        edge_attr_connected = self._graph.edge_attr[connected_edges_mask, :]

        if idx_new.numel() > 0:
            edges_new = torch.nonzero(connected_node_mask).T
            edges_new[1, :] = idx_new[edges_new[1, :]]
            edges_new_inv = torch.stack([edges_new[1, :], edges_new[0, :]], dim=0)
            edges_new = torch.cat([edges_new, edges_new_inv], dim=1)
            edges_new = torch.unique(edges_new, dim=1)   # rm doubled edges from concatenating the inverse

            edges_new, _ = remove_self_loops(edges_new)

            graph_new = Data(x=x_all, pos=pos_all, edge_index=edges_new)
            edge_attr_new = self.__edge_attr(graph_new).edge_attr

            edge_index = torch.cat([edges_connected, edges_new], dim=1)
            edge_attr = torch.cat([edge_attr_connected, edge_attr_new])
        else:
            edge_index = edges_connected
            edge_attr = edge_attr_connected

        x_j = x_all[edge_index[0, :], :]
        phi = self.message(x_j, edge_attr=edge_attr)

        y_update = self.aggregate(phi, index=edge_index[1, :], ptr=None, dim_size=x_all.size()[0])

        y = torch.cat([self._graph.y.clone(), torch.zeros(x_new.size()[0], self.out_channels, device=x.device)])
        y[idx_update] = y_update[idx_update]
        return y

    #####################################################################################
    # Synchronous updates ###############################################################
    #####################################################################################
    def _sync_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                      ) -> torch.Tensor:
        if self.__is_initial:
            x = torch.cat([self._graph.x, x], dim=0)
            pos = torch.cat([self._graph.pos, pos], dim=0)

        if edge_attr is None:
            if edge_index is None:
                edge_index = self._compute_edges(pos)
            attr_data = Data(pos=pos, edge_index=edge_index)
            edge_attr = self.__edge_attr(attr_data).edge_attr

        return SplineConv.forward(self, x, edge_index=edge_index, edge_attr=edge_attr)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update(out, **update_kwargs)
