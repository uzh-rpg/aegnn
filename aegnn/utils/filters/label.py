import torch_geometric
from typing import List, Union

from .base import Filter


class Label(Filter):
    """Filter data objects by a predefined set of labels. If the `label` attribute of the data object is
    element of the pre-defined set, let it through, otherwise reject it.

    Args:
        labels: list of the selected label (or ":"-separated string).
    """

    def __init__(self, labels: Union[List[str], str]):
        self.labels = set(labels if type(labels) == list else labels.split(":"))

    def __call__(self, data: torch_geometric.data.Data) -> bool:
        if hasattr(data, "label"):
            label = getattr(data, "label")
            if type(label) == str:
                return label in self.labels
            elif type(label) == list:
                return not set(label).isdisjoint(self.labels)
            else:
                raise NotImplementedError(f"Label filter not implemented for attribute type {type(label)}")
        return True

    def __repr__(self):
        labels_sorted = ":".join(sorted(list(self.labels)))
        return f"{self.__class__.__name__}[labels={labels_sorted}]"
