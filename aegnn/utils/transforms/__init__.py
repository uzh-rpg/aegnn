from aegnn.utils.transforms.fsf import FSF
from aegnn.utils.transforms.nvst import NVST
from aegnn.utils.transforms.uniform import Uniform

################################################################################################
# Access functions #############################################################################
################################################################################################
import typing
from .base import Transform


def by_name(name: str, **kwargs) -> typing.Union[Transform, None]:
    from ..io import select_by_name
    return select_by_name([FSF, NVST, Uniform], name=name, **kwargs)


def from_description(description: str) -> typing.Union[Transform, None]:
    from ..io import parse_description
    if description.lower() == "none":
        return None
    name, kwargs = parse_description(description)
    return by_name(name, **kwargs)
