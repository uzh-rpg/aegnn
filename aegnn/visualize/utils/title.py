import torch
from typing import Any


def make_title(title: Any, default: Any) -> str:
    def readable_string(x: Any) -> str:
        if type(x) == str:
            return x
        elif type(x) == torch.Tensor:
            return str(round(float(x), 2))
        elif type(x) == float:
            return str(round(x, 2))
        else:
            return x

    if title is not None:
        return readable_string(title)
    return readable_string(default)
