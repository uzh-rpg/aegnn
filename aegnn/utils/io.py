import os
from typing import Union, Any, Tuple, Dict


def select_by_name(choices, name: str, **kwargs) -> Union[Any, None]:
    if name is None or name == "none":
        return None

    name = name.lower()
    for c in choices:
        c_name = c.__name__.lower()
        if name == c_name:
            return c(**kwargs)

    else:
        raise ValueError(f"Selection with name {name} is not known!")


def parse_description(description: str) -> Tuple[str, Dict[str, Any]]:
    name = description.split("[")[0]

    args = description[description.find("[") + 1:description.find("]")]
    kwargs = {}
    for argument in args.split(","):
        key, value = argument.split("=")
        key = key.replace(" ", "")
        kwargs[key] = value

    return name, kwargs


def setup_environment(env_dict: Dict[str, str]):
    for key, value in env_dict.items():
        if key not in os.environ:
            os.environ[key] = value
