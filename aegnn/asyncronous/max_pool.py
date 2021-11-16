import logging
import math
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid
from torch_geometric.nn.pool.pool import pool_edge
from torch_geometric.typing import Adj
from torch_scatter import scatter_max, scatter_sum
from typing import List

from aegnn.models.layer import MaxPooling
from .base.base import add_async_graph, async_context
from .base.utils import compute_edges, graph_new_nodes, graph_changed_nodes
from .flops import compute_flops_voxel_grid


def __intersection(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the unique common elements in x and y, with x and y being in generally different shape.
    Attention: Only works for integer typed tensors x & y. """
    assert x.dtype == y.dtype == torch.long
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor([], device=x.device)

    xy_max = max(torch.max(x), torch.max(y))
    x_array = torch.zeros(xy_max + 1, device=x.device)
    y_array = torch.zeros(xy_max + 1, device=y.device)
    x_array[x] = 1
    y_array[y] = 1
    z_array = torch.mul(x_array, y_array)
    return torch.flatten(torch.nonzero(z_array))


def __dense_process(module: MaxPooling, x: torch.Tensor, pos: torch.Tensor, batch: torch.LongTensor = None,
                    edge_index: Adj = None) -> Data:
    batch = torch.zeros(x.size()[0], device=x.device)  # no batch processing
    if edge_index is None:
        edge_index = compute_edges(module, pos=pos)
    clusters = __get_clusters(module, pos=pos)

    graph = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
    graph_coarse = max_pool(clusters, data=graph, transform=module.transform)
    graph_coarse.vdx = torch.unique(clusters, sorted=True)
    return graph_coarse


def __graph_initialization(module: MaxPooling, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None) -> Data:
    """Graph initialization for asynchronous update.

    Both the input as well as the output graph have to be stored, in order to avoid repeated computation. The
    input graph is used for spotting changed or new nodes (as for other asyn. layers), while the output graph
    is compared to the set of diff & new nodes, in order to be updated. Depending on the type of pooling (max, mean,
    average, etc) not only the output voxel feature have to be stored but also aggregations over all nodes in
    one output voxel such as the sum or count.

    Next to the features the node positions are averaged over all nodes in the voxel, as well. To do so,
    position aggregations (count, sum) are stored and updated, too.
    """
    logging.debug(f"Input graph with x = {x.shape} and pos = {pos.shape}")
    graph_out = __dense_process(module, x, pos, edge_index=None)
    module.asy_graph = Data(x=x, pos=pos)
    logging.debug(f"Resulting in coarse graph {graph_out}")

    # Compute the voxel index for every node (clustering), and determine the max. feature vector over
    # all nodes that are assigned to the same voxel, independently for all dimensions. Example:
    # x1 = [1, 3, 4, 5, 1], x2 = [0, 2, 6, 10, 20] => x_voxel = [1, 3, 6, 10, 20]
    module.asy_graph.vdx = __get_clusters(module, pos=pos)  # voxel index of every node
    _, index = torch.unique(module.asy_graph.vdx, sorted=True, return_inverse=True)
    _, argmax = scatter_max(x, index=index, dim=0)

    # Store all of the nodes that contribute to the voxel max feature vector in at least one dimension.
    argmax_nodes = torch.unique(argmax)
    module.asy_node_max_index = torch.flatten(argmax_nodes).long()

    # Store aggregations and final coarse (output) graph.
    module.asy_voxel_pos_sum = scatter_sum(pos, index=index, dim=0)
    module.asy_voxel_node_count = scatter_sum(torch.ones_like(module.asy_graph.vdx, device=x.device), index=index)
    module.asy_graph_coarse = graph_out.clone()
    assert module.asy_voxel_pos_sum.shape[0] == module.asy_graph_coarse.num_nodes
    assert module.asy_voxel_node_count.shape[0] == module.asy_graph_coarse.num_nodes

    # Compute number of floating point operations (no cat, flatten, etc.).
    if module.asy_flops_log is not None:
        flops = compute_flops_voxel_grid(pos)
        flops += graph_out.x.numel() + graph_out.edge_index.numel()  # every edge has to be re-assigned
        flops += 3 * index.numel()  # scatter with index
        module.asy_flops_log.append(flops)
    return graph_out


def __graph_process(module: MaxPooling, x: torch.Tensor, pos: torch.Tensor, edge_index: Adj = None) -> Data:
    pos = module.asy_pos  # input pos is pos of new event, not whole graph due to dense & sparse code sharing
    _, diff_idx = graph_changed_nodes(module, x=x)
    _, new_idx = graph_new_nodes(module, x=x)
    logging.debug(f"Subgraph contains {new_idx.numel()} new and {diff_idx.numel()} diff nodes")

    # Compute the intersection between nodes that have been changed and nodes that contribute to the
    # voxel's feature vector (max values).
    replaced_idx = __intersection(diff_idx.long(), module.asy_node_max_index)
    logging.debug(f"... from which {replaced_idx.numel()} nodes contributed to the coarse graph")

    # As not all of the feature values, that do not contribute to the voxel feature vector (max vector), are
    # stored, when one of the contributing nodes has changed, the voxel feature vector has to be re-evaluated
    # by looking at all (!) of the nodes that are assigned to the voxel. Therefore, for every changed index
    # add all of the nodes in the same voxel to the list of nodes of the subgraph to look at, and reset the
    # nodes aggregations.
    graph_vdx = getattr(module.asy_graph, "vdx")
    for idx in replaced_idx:
        nodes_voxel = torch.nonzero(torch.eq(graph_vdx, graph_vdx[idx]))[:, 0]
        diff_idx = torch.cat([diff_idx, nodes_voxel])

        voxel_idx = module.asy_graph.vdx[idx]
        coarse_idx = torch.eq(getattr(module.asy_graph_coarse, "vdx"), voxel_idx)

        module.asy_graph_coarse.x[coarse_idx] = -999999
        module.asy_voxel_pos_sum[coarse_idx] -= pos[idx]
        module.asy_voxel_node_count[coarse_idx] -= 1

    update_idx = torch.cat([diff_idx, new_idx])
    x_update = x[update_idx, :]
    pos_update = pos[update_idx, :]

    # Max/Average pool the features x and positions respectively.
    vdx_update = __get_clusters(module, pos=pos_update)
    x_scatter = torch.cat([x_update, module.asy_graph_coarse.x])
    node_index_scatter = torch.cat([update_idx, module.asy_node_max_index])
    pos_sum_scatter = torch.cat([pos_update, module.asy_voxel_pos_sum])
    node_count_scatter = torch.cat([torch.ones_like(update_idx, device=x.device), module.asy_voxel_node_count])
    clusters = torch.cat([vdx_update, getattr(module.asy_graph_coarse, "vdx")])

    clusters_unique, index = torch.unique(clusters, sorted=True, return_inverse=True)
    x_max, argmax = scatter_max(x_scatter, index=index, dim=0)

    voxel_pos_sum = scatter_sum(pos_sum_scatter, index=index, dim=0)
    voxel_node_count = scatter_sum(node_count_scatter, index=index)
    pos_mean = torch.div(voxel_pos_sum, voxel_node_count.view(-1, 1))  # index is sorted, so output is too

    # The coarsened graph is reconnected by dropping all edges inside a voxel and
    # unifying all edges between voxels.
    vdx = torch.cat([getattr(module.asy_graph, "vdx"), vdx_update[diff_idx.numel():]])  # add new node index
    vdx[diff_idx] = vdx_update[:diff_idx.numel()]  # change diff node index
    edges_coarse = torch.empty((2, 0), dtype=torch.long, device=x.device)
    if edge_index is not None:
        edges_coarse, _ = pool_edge(cluster=vdx, edge_index=edge_index, edge_attr=None)

    # Create the coarsened graph and update the internal graph. While the coarsened graph can be
    # overwritten completely, as it has been re-computed, most elements of the un-coarsened graph
    # are unchanged, and therefore only have to be partly updated.
    graph_out = Data(x=x_max, pos=pos_mean, edge_index=edges_coarse)
    module.asy_graph = Data(x=x, pos=pos, edge_index=edge_index, vdx=vdx).clone()

    module.asy_node_max_index = torch.flatten(node_index_scatter[argmax]).long()
    module.asy_voxel_pos_sum = voxel_pos_sum
    module.asy_voxel_node_count = voxel_node_count
    module.asy_graph_coarse = Data(x=graph_out.x, pos=graph_out.pos, edge_index=edges_coarse, vdx=clusters_unique)

    # Compute number of floating point operations (no cat, flatten, etc.).
    if module.asy_flops_log is not None:
        flops = x_scatter.size()[0] + pos_sum_scatter.numel() + node_count_scatter.numel()  # pooling
        flops += voxel_pos_sum.numel()  # pos mean
        module.asy_flops_log.append(flops)
    # For asychronous processing we assume that all events are in the same "batch".
    graph_out.batch = torch.zeros(graph_out.num_nodes, dtype=torch.long, device=graph_out.x.device)

    # Max pooling coarsens the graph, so the pos vector of all subsequent layer has to be updated..
    module.asy_pass_attribute('asy_pos', graph_out.pos)
    return graph_out


def __get_clusters(module, pos: torch.Tensor) -> torch.LongTensor:
    num_pos, num_dims = pos.shape
    grid_start = torch.zeros(num_dims, device=pos.device)
    grid_end = module.grid_size
    return voxel_grid(pos, batch=torch.zeros(num_pos), size=module.voxel_size, start=grid_start, end=grid_end)


def __get_num_voxels(module, dim: int = None) -> int:
    num_dims = len(module.voxel_size)
    num_voxels = [int(module.grid_size[i] / module.voxel_size[i]) + 1 for i in range(num_dims)]
    if dim is not None:
        return num_voxels[dim]
    return math.prod(num_voxels)


def make_max_pool_asynchronous(module: MaxPooling, grid_size: List[int], r: float,
                               log_flops: bool = False, log_runtime: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for graph max pooling layer.
    By overwriting parts of the module asynchronous processing can be enabled without the need re-creating the
    object. So, a max pooling layer can be converted by, for example:

    ```
    module = MaxPool([4, 4])
    module = make_max_pool_asynchronous(module)
    ```

    :param module: standard max pooling module.
    :param grid_size: grid size (grid starting at 0, spanning to `grid_size`), >= `size`.
    :param r: update radius around new events.
    :param log_flops: log flops of asynchronous update.
    :param log_runtime: log runtime of asynchronous update.
    """
    assert hasattr(module, "voxel_size")
    assert len(module.voxel_size) == len(grid_size)
    assert all([module.voxel_size[i] <= grid_size[i] for i in range(len(module.voxel_size))])
    assert all([grid_size[i] % module.voxel_size[i] == 0 for i in range(len(module.voxel_size))])

    module = add_async_graph(module, r=r, log_flops=log_flops, log_runtime=log_runtime)
    module.asy_pos = None
    module.asy_graph_coarse = None  # coarse output graph
    module.asy_node_max_index = None  # index of max. node in input data
    module.asy_voxel_pos_sum = None  # sum of positions per voxel
    module.asy_voxel_node_count = None  # count of nodes per voxel

    module.grid_size = grid_size  # grid size in N dimensions

    def async_forward(x: torch.Tensor, pos: torch.Tensor = None,
                      batch=None, edge_index: Adj = None, return_data_obj: bool = False):
        with async_context(module, __graph_initialization, __graph_process) as func:
            data_out = func(module, x=x, pos=pos, edge_index=edge_index)

        # If defined, apply transform to output data.
        if module.transform is not None:
            data_out = module.transform(data_out)

        # Following the convention defined in `aegnn.models.layer.max_pool`, forward either returns a data object
        # or its parts (x, pos, batch, edge_index, edge_attr)
        return data_out if return_data_obj else \
            (data_out.x, data_out.pos, getattr(data_out, "batch"), data_out.edge_index, data_out.edge_attr)

    module.forward = async_forward
    return module
