import torch


def intersection(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the unique common elements in x and y, with x and y being in generally different shape.
    Attention: Only works for integer typed tensors x & y. """
    assert x.dtype == y.dtype == torch.long
    xy_max = max(torch.max(x), torch.max(y))
    x_array = torch.zeros(xy_max + 1, device=x.device)
    y_array = torch.zeros(xy_max + 1, device=y.device)
    x_array[x] = 1
    y_array[y] = 1
    z_array = torch.mul(x_array, y_array)
    return torch.flatten(torch.nonzero(z_array))
