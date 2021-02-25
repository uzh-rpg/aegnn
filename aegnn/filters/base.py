import torch_geometric


class Filter:

    def __call__(self, data: torch_geometric.data.Data) -> bool:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
