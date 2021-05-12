import pytest
import torch

from aegnn.asyncronous import AsyMaxPool
from torch_geometric.data import Data
from typing import List


class TestMaxPool:

    @staticmethod
    def evaluate(data: Data, data_new: Data, size: List[int], grid_size: List[int], r: float = 3.0):
        max_pool = AsyMaxPool(size, grid_size=grid_size, r=r, transform=None)
        _ = max_pool.forward(data.x, data.pos)
        out_sparse = max_pool.forward(data_new.x, data_new.pos)

        max_pool.synchronous()
        out_dense = max_pool.forward(data_new.x, data_new.pos)
        return out_sparse, out_dense

    def test_changing_event(self):
        x = torch.tensor([4, 3, 5, 2, 1]).view(-1, 1)
        pos = torch.tensor([[2, 1], [6, 2], [6, 1], [5, 2], [7, 7]]).view(-1, 2)
        data = Data(x=x, pos=pos)

        x_new = data.x.clone()
        x_new[2, :] = 1
        data_new = Data(x=x_new, pos=data.pos.clone())

        out_sparse, out_dense = self.evaluate(data, data_new, size=[4, 4], grid_size=[10, 10])
        diff = torch.nonzero(torch.sum(out_dense.x - out_sparse.x, dim=1).abs() > 1e-3)
        assert diff.numel() == 0
