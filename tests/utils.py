import numpy as np
import torch

from aegnn.utils.tensors import intersection


def test_intersection():
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([4, 4, 1, 9, 11, 41, 0])

    # Compute the intersection indices in the implemented way.
    idx = intersection(x, y).numpy().tolist()

    # Compute the intersection using python sets (inefficient, but accurate).
    x_set = x.numpy()
    y_set = y.numpy()
    idx2 = list(set(x_set) & set(y_set))
    assert sorted(idx) == sorted(idx2)
