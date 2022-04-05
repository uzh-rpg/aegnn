import torch
import torch_geometric

from typing import List


def compute_flops_conv(module, idx_new: List[int], idx_diff: List[int], edges: torch.LongTensor) -> int:
    assert type(idx_new) == type(idx_diff) == list
    flops = 0
    node_idx_unique, edge_count = torch.unique(edges, return_counts=True)
    node_idx_unique = node_idx_unique.detach().cpu().numpy()  # can be slow, just for flops evaluation
    edge_count = edge_count.detach().cpu().numpy()
    edge_count_dict = {ni: c for ni, c in zip(node_idx_unique, edge_count)}

    # Iterate over every different and every new node, and add the number of flops introduced
    # by the node to the overall flops count of the layer.
    for i in idx_new + idx_diff:
        if i not in edge_count_dict.keys():
            continue  # no edges from this node
        num_neighs_i = edge_count_dict[i]
        flops += __compute_flops_node(module, num_neighs=num_neighs_i)
    return flops


def __compute_flops_node(module, num_neighs: int) -> int:
    ni = num_neighs  # to use the same notation as in the derivation
    m_in = module.in_channels
    m_out = module.out_channels

    if isinstance(module, torch_geometric.nn.conv.SplineConv):
        nm = module.dim
        np = module.weight.size()[0]
        d = module.degree
        return ni * m_out * m_in * (1 + np) + ni * (2 * d + 2 * nm * d - 1)
    else:
        module_type = type(module).__name__
        raise NotImplementedError(f"FLOPS computation not implemented for module type {module_type}")
