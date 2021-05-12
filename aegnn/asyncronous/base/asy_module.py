import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.typing import Adj
from typing import Tuple


class AsyModule:

    def __init__(self, r: float, do_graph_updates: bool = True):
        self.do_graph_updates = do_graph_updates

        self._graph = None  # activations & prior events
        self._radius = r  # update radius around new events
        self.__is_async = None  # sync. or asynchronous update
        self.__processing_func = None  # processing function (depending on whether layer is initial & async)

        self.asynchronous()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None):
        if self._graph is None:
            return self.graph_initialization(x=x, pos=pos)
        return self.__processing_func(x=x, pos=pos)

    def _async_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                       ) -> torch.Tensor:
        raise NotImplementedError

    def _sync_process(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr: torch.Tensor = None
                      ) -> torch.Tensor:
        raise NotImplementedError

    #####################################################################################
    # Graph Operations ##################################################################
    #####################################################################################
    def graph_initialization(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None, edge_attr=None):
        raise NotImplementedError

    def _graph_changed_nodes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        num_prev_nodes = self._graph.num_nodes
        x_graph = x[:num_prev_nodes]
        different_node_idx = torch.nonzero(x_graph - self._graph.x)[:, 0]
        return x_graph, different_node_idx

    def _graph_new_nodes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        num_prev_nodes = self._graph.num_nodes
        assert x.size()[0] >= num_prev_nodes, "node deletion is not supported"
        x_graph = x[num_prev_nodes:]
        new_node_idx = torch.arange(num_prev_nodes, x.size()[0], device=x.device, dtype=torch.long).long()
        return x_graph, new_node_idx

    def _compute_edges(self, pos: torch.Tensor) -> torch.LongTensor:
        return radius_graph(pos, r=self._radius, max_num_neighbors=pos.size()[0])

    #####################################################################################
    # Mode and Forward ##################################################################
    #####################################################################################
    def synchronous(self):
        self.__is_async = False
        self.__processing_func = self._sync_process

    def asynchronous(self):
        self.__is_async = True
        self.__processing_func = self._async_process

    #####################################################################################
    # Read-Only Properties ##############################################################
    #####################################################################################
    @property
    def is_async(self) -> bool:
        return self.__is_async

    @property
    def graph(self) -> Data:
        return self._graph
