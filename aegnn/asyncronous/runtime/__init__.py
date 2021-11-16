import logging


def compute_runtime_from_module(module) -> float:
    """Compute runtime from a GNN module (after the forward pass).
    :param module: module to infer the runtime from.
    """
    module_name = module.__class__.__name__

    if hasattr(module, "asy_runtime_log") and module.asy_runtime_log is not None:
        assert type(module.asy_runtime_log) == list, "asyc. runtime log must be a list"
        assert len(module.asy_runtime_log) > 0, "asynchronous runtime log is empty"
        runtime = module.asy_runtime_log[-1]

    else:
        logging.debug(f"Module {module_name} has no runtime log, using runtime = 0s")
        return 0

    logging.debug(f"Module {module_name} adds {runtime}s")
    return runtime
