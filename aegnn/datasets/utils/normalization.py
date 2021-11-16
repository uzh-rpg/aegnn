import torch


def normalize_time(ts: torch.Tensor, beta: float = 0.5e-5) -> torch.Tensor:
    """Normalizes the temporal component of the event pos by using beta re-scaling

    :param ts: time-stamps to normalize in microseconds [N].
    :param beta: re-scaling factor.
    """
    return (ts - torch.min(ts)) * beta
