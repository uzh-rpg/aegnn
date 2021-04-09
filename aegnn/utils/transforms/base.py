import torch_geometric


class Transform:

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
