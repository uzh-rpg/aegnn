import os
from typing import List

from .os import path_local_to_global


def read_txt_to_list(file_path: str, is_local: bool = True, default: List[str] = None) -> List[str]:
    """Read text file to list of strings (by line).

    :param file_path: path to yaml file that should be read.
    :param is_local: local or global file path.
    :param default: default return if the file does not exist.
    :return: contained rows.
    """
    if is_local:
        file_path = path_local_to_global(local_path=file_path)
    if not os.path.isfile(file_path):
        return default
    with open(file_path, "r") as file:
        return file.read().splitlines()


def write_list_to_file(data: List[str], file_path: str):
    """Write all elements in a list to a file, each element in a new line.

    :param data: list of strings to write.
    :param file_path: path to output file (over-write).
    """
    data_string = "\n".join(data)
    with open(file_path, "w+") as f:
        f.write(data_string)
        f.close()
