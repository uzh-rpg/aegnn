import random
import torch_geometric
import torch_geometric.transforms as T

from torch_geometric.transforms import Cartesian

from .base import Transform


class FSF(Transform):

    def __init__(self, p1: float = 0.3, s2: float = 0.8, p3: float = 0.2, seed: int = 12345):
        random.seed(seed)
        self.tf = T.Compose([T.RandomFlip(axis=0, p=p1), T.RandomScale((s2, 0.999)), T.RandomFlip(axis=1, p=p3)])
        self.edge_attr = Cartesian(cat=False)

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        data = self.tf(data)

        x_min, y_min = data.pos.min(dim=0).values
        x_max, y_max = data.pos.max(dim=0).values
        data.pos[:, 0] = (data.pos[:, 0] - x_min) / (x_max - x_min) * 239
        data.pos[:, 1] = (data.pos[:, 1] - y_min) / (y_max - y_min) * 179

        data = self.edge_attr(data)
        return data

    def __repr__(self):
        name = self.__class__.__name__
        tfs = self.tf.transforms
        return f"{name}[p0={tfs[0].p}, s2={tfs[1].scales[0]}, p3={tfs[2].p}]"
