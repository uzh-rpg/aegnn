from .graph_res import GraphRes
from .graph_wen import GraphWen

################################################################################################
# Access functions #############################################################################
################################################################################################
import torch


def by_name(name: str) -> torch.nn.Module.__class__:
    if name == "graph_res":
        return GraphRes
    elif name == "graph_wen":
        return GraphWen
    else:
        raise NotImplementedError(f"Network {name} is not implemented!")
