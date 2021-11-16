import logging

import torch.nn
import torch_geometric

from torch_geometric.nn.norm import BatchNorm
from aegnn.models.layer import MaxPooling

from aegnn.asyncronous.conv import make_conv_asynchronous
from aegnn.asyncronous.batch_norm import make_batch_norm_asynchronous
from aegnn.asyncronous.linear import make_linear_asynchronous
from aegnn.asyncronous.max_pool import make_max_pool_asynchronous

from aegnn.asyncronous.flops import compute_flops_from_module
from aegnn.asyncronous.runtime import compute_runtime_from_module
from aegnn.asyncronous.base.callbacks import CallbackFactory


def make_model_asynchronous(module, r: float, grid_size=None, edge_attributes=None,
                            log_flops: bool = False, log_runtime: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for graph convolutional layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a convolutional layer can be converted by, for example:

    ```
    module = GCNConv(1, 2)
    module = make_conv_asynchronous(module)
    ```

    :param module: convolutional module to transform.
    :param grid_size: grid size (grid starting at 0, spanning to `grid_size`), >= `size` for pooling operations,
                      e.g. the image size.
    :param r: update radius around new events.
    :param edge_attributes: function for computing edge attributes (default = None), assumed to be the same over
                            all convolutional layers.
    :param log_flops: log flops of asynchronous update.
    :param log_runtime: log runtime of asynchronous update.
    """
    assert isinstance(module, torch.nn.Module), "module must be a `torch.nn.Module`"
    conv_is_initial = True
    model_forward = module.forward
    module.asy_flops_log = [] if log_flops else None
    module.asy_runtime_log = [] if log_runtime else None
    callback_keys = []

    # Make all layers asynchronous that have an implemented asynchronous function. Otherwise use
    # the synchronous forward function.
    log_kwargs = dict(log_flops=log_flops, log_runtime=log_runtime)
    for key, nn in module._modules.items():
        nn_class_name = nn.__class__.__name__
        logging.debug(f"Making layer {key} of type {nn_class_name} asynchronous")

        if nn_class_name in torch_geometric.nn.conv.__all__:
            module._modules[key] = make_conv_asynchronous(nn, r=r, edge_attributes=edge_attributes,
                                                          is_initial=conv_is_initial, **log_kwargs)
            conv_is_initial = False
            callback_keys.append(key)

        elif isinstance(nn, MaxPooling):
            assert grid_size is not None, "grid size must be defined for pooling operations"
            module._modules[key] = make_max_pool_asynchronous(nn, grid_size=grid_size, r=r, **log_kwargs)
            callback_keys.append(key)

        elif isinstance(nn, BatchNorm):
            module._modules[key] = make_batch_norm_asynchronous(nn, **log_kwargs)
            # no callbacks required

        elif isinstance(nn, torch.nn.Linear):
            module._modules[key] = make_linear_asynchronous(nn, **log_kwargs)
            callback_keys.append(key)

        else:
            logging.debug(f"Asynchronous module for {nn_class_name} is not implemented, using dense module.")

    # Set callbacks for overwriting attributes on subsequent network layers, from a function factory design.
    callback_index = 0
    cb_listeners = [module._modules[key] for key in callback_keys]
    for key, nn in module._modules.items():
        if key not in callback_keys or callback_index >= len(callback_keys) - 1:
            continue
        nn.asy_pass_attribute = CallbackFactory(cb_listeners[callback_index + 1:], log_name=nn.__repr__())
        callback_index += 1
    module.asy_pass_attribute = CallbackFactory(cb_listeners, log_name="base model")

    def async_forward(data: torch_geometric.data.Data, *args, **kwargs):
        module.asy_pass_attribute('asy_pos', data.pos)
        out = model_forward(data, *args, **kwargs)

        if module.asy_flops_log is not None:
            flops_count = [compute_flops_from_module(layer) for layer in module._modules.values()]
            module.asy_flops_log.append(sum(flops_count))
            logging.debug(f"Model's modules update with overall {sum(flops_count)} flops")
        if module.asy_runtime_log is not None:
            runtimes = [compute_runtime_from_module(layer) for layer in module._modules.values()]
            module.asy_runtime_log.append(sum(runtimes))
            logging.debug(f"Model's modules took overall {sum(runtimes)}s")
        return out

    module.forward = async_forward
    return module


__all__ = [
    "make_conv_asynchronous",
    "make_linear_asynchronous",
    "make_max_pool_asynchronous",
    "make_model_asynchronous"
]