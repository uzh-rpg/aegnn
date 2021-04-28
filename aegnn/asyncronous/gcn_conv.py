import torch

from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import Data
from torch_geometric.typing import Adj

from .base import AsyModule


class AsyGCNConv(GCNConv, AsyModule):

    def __init__(self, in_channels: int, out_channels: int, r: float,
                 do_graph_updates: bool = True, is_initial: bool = False, **kwargs):
        GCNConv.__init__(self, in_channels, out_channels=out_channels, aggr="add", normalize=False, **kwargs)
        AsyModule.__init__(self, do_graph_updates=do_graph_updates, r=r)
        self.__is_initial = is_initial

    #####################################################################################
    # Graph operations ##################################################################
    #####################################################################################
    def graph_initialization(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr=None):
        if edge_index is None:
            edge_index = self._compute_edges(pos)
        y = GCNConv.forward(self, x, edge_index=edge_index)
        self._graph = Data(x=x, pos=pos, edge_index=edge_index, y=y)
        return self._graph.y

    def forward(self, x: torch.Tensor, pos: torch.Tensor, **kwargs):
        return AsyModule.forward(self, x, pos, **kwargs)

    #####################################################################################
    # Asynchronous updates ##############################################################
    #####################################################################################
    def _async_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                       ) -> torch.Tensor:
        # Update the nodes that already are in the internal graph, but have a different activation.
        # Therefore, search them and completely re-calculate their activations based on their neighborhood.
        if not self.__is_initial:
            edge_i, edge_j = self._graph.edge_index
            x_diff, diff_idx = self._graph_changed_nodes(x)

            # TODO: vectorize implementation using a node mask??
            for node_idx in diff_idx:
                edges_idx = torch.nonzero(torch.eq(edge_j, node_idx))[:, 0]
                neigh_idx = edge_i[edges_idx]
                x_diff = torch.matmul((x_diff[node_idx, :] - self._graph.x[node_idx, :]), self.weight)
                self._graph.y[neigh_idx, :] += x_diff

        # Find the nodes that are connected to the new node (`event`), by computing all the distances between the
        # nodes in the previous graph and the new nodes. Nodes within some radius `r` around the event will be updated.
        # Add the event's activation as the sum of the weighted features of surrounding events.
        # WARNING: The `connected_node_mask` only includes the distances from new nodes to the graph nodes, not the
        #          between new ones. As a consequence edges between new nodes are not considered.
        if not self.__is_initial:
            x_new, new_idx = self._graph_new_nodes(x)
            pos_new = pos[new_idx, :]
        else:
            x_new, pos_new, new_idx = x, pos, torch.arange(0, x.size()[0], device=x.device)
        connected_node_mask = (torch.cdist(self._graph.pos, pos_new) <= self._radius)

        # Update the masked nodes by adding the weighted event feature. This is only valid if the aggregation
        # is "adding", no normalization occurs and the radius graph allows for infinite neighbors, as otherwise
        # the neighborhood structure would change and thus more nodes would have to be updated.
        phi = torch.matmul((x_new.float().T * connected_node_mask).unsqueeze(dim=-1), self.weight)
        self._graph.y += torch.sum(phi, dim=1)
        phi = torch.matmul((connected_node_mask * self._graph.x).unsqueeze(dim=-1), self.weight)
        y_new = torch.sum(phi, dim=0)

        # Update the graph for the next iteration.
        y_output = torch.cat([self._graph.y, y_new], dim=0)
        if self.do_graph_updates:
            for i, idx in enumerate(new_idx):
                neigh_i = torch.flatten(torch.nonzero(connected_node_mask[:, i]))
                new_edges_i = torch.stack([neigh_i, torch.ones_like(neigh_i) * idx])
                self._graph.edge_index = torch.cat([self._graph.edge_index, new_edges_i], dim=1)

            self._graph.x = torch.cat([self._graph.x, x_new], dim=0)
            self._graph.pos = torch.cat([self._graph.pos, pos_new], dim=0)
            self._graph.y = y_output
        return y_output

    #####################################################################################
    # Synchronous updates ###############################################################
    #####################################################################################
    def _sync_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                      ) -> torch.Tensor:
        if self.__is_initial:
            x = torch.cat([self._graph.x, x], dim=0)
            pos = torch.cat([self._graph.pos, pos], dim=0)
        if edge_index is None:
            edge_index = self._compute_edges(pos)
        return GCNConv.forward(self, x, edge_index=edge_index)
