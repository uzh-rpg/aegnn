import torch_geometric
from typing import List, Union

from .base import Filter


class ClassID(Filter):
    """Filter data objects by a predefined set of class_ids. If the `class_id` attribute of the data object is
    element of the pre-defined set, let it through, otherwise reject it."""

    def __init__(self, classes: Union[List[str], str]):
        self.classes = classes if type(classes) == list else classes.split(":")

    def __call__(self, data: torch_geometric.data.Data) -> bool:
        return data.__getattribute__("class_id") in self.classes

    def __repr__(self):
        classes_sorted = ":".join(sorted(self.classes))
        return f"{self.__class__.__name__}[classes={classes_sorted}]"
