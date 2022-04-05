from contextlib import contextmanager
import logging
import time


def add_async_graph(module, r: float = None, log_flops: bool = False, log_runtime: bool = False):
    module.asy_graph = None
    module.asy_flops_log = [] if log_flops else None
    module.asy_runtime_log = [] if log_runtime else None
    if r is not None:
        module.asy_radius = r
    return module


def make_asynchronous(module, initialization_func, processing_func):
    def async_forward(*args, **kwargs):
        with async_context(module, initialization_func, processing_func) as func:
            output = func(module, *args, **kwargs)
        return output

    module.forward = async_forward
    return module


@contextmanager
def async_context(module, initialization_func, processing_func):
    do_log_runtime = getattr(module, "asy_runtime_log", None) is not None
    start_time = time.time() if do_log_runtime else None

    if module.asy_graph is None:
        logging.debug(f"Graph initialization of module {module}")
        yield initialization_func
    else:
        logging.debug(f"Calling processing of module {module}")
        yield processing_func

    if do_log_runtime:
        dt = time.time() - start_time
        module.asy_runtime_log.append(dt)
