import random
import torch_geometric
import torch_geometric.transforms as T

from .base import Transform


class FSF(Transform):

    def __init__(self, p1: float = 0.3, s2: float = 0.95, p3: float = 0.2, seed: int = 12345):
        random.seed(seed)
        self.tf = T.Compose([T.RandomFlip(axis=0, p=p1), T.RandomScale((s2, 0.999)), T.RandomFlip(axis=1, p=p3)])

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        return self.tf(data)

    def __repr__(self):
        name = self.__class__.__name__
        tfs = self.tf.transforms
        return f"{name}[p0={tfs[0].p}, s2={tfs[1].scales[0]}, p3={tfs[2].p}]"
