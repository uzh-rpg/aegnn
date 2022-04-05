import numpy as np
import torch
import torch_geometric

from torch_geometric.nn.norm import BatchNorm
from .base.base import make_asynchronous, add_async_graph


def __graph_initialization(module: BatchNorm, x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=0)
    var = torch.var(x, dim=0) + module.module.eps
    y = (x - mean) / var * module.module.weight + module.module.bias
    module.asy_graph = torch_geometric.data.Data(x=mean, variance=var)

    # If required, compute the flops of the asynchronous update operation.
    # flops computation from https://github.com/sovrasov/flops-counter.pytorch/
    if module.asy_flops_log is not None:
        flops = int(np.prod(x.size()) * y.size()[-1])
        module.asy_flops_log.append(flops)
    return y


def __graph_processing(module: BatchNorm, x: torch.Tensor) -> torch.Tensor:
    """Batch norms only execute simple normalization operation, which already is very efficient. The overhead
    for looking for diff nodes would be much larger than computing the dense update.

    However, a new node slightly changes the feature distribution and therefore all activations, when calling
    the dense implementation. Therefore, we approximate the distribution with the initial distribution as
    num_new_events << num_initial_events.
    """
    y = (x - module.asy_graph.x) / module.asy_graph.variance * module.module.weight + module.module.bias

    # If required, compute the flops of the asynchronous update operation.
    if module.asy_flops_log is not None:
        flops = int(x.shape[0] * x.shape[1]) * 4
        module.asy_flops_log.append(flops)
    return y


def __check_support(module):
    return True


def make_batch_norm_asynchronous(module: BatchNorm, log_flops: bool = False, log_runtime: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for batch norm (1d) layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a layer can be converted by, for example:

    ```
    module = BatchNorm(4)
    module = make_batch_norm_asynchronous(module)
    ```

    :param module: batch norm module to transform.
    :param log_flops: log flops of asynchronous update.
    :param log_runtime: log runtime of asynchronous update.
    """
    assert __check_support(module)
    module = add_async_graph(module, r=None, log_flops=log_flops, log_runtime=log_runtime)
    return make_asynchronous(module, __graph_initialization, __graph_processing)
