from aegnn.utils.filters.class_id import Label

################################################################################################
# Access functions #############################################################################
################################################################################################
import typing
from .base import Filter


def by_name(name: str, **kwargs) -> typing.Union[Filter, None]:
    from ..io import select_by_name
    choices = [Label]
    return select_by_name(choices, name=name, **kwargs)


def from_description(description: str) -> typing.Union[Filter, None]:
    from ..io import parse_description
    if description.lower() == "none":
        return None
    name, kwargs = parse_description(description)
    return by_name(name, **kwargs)
