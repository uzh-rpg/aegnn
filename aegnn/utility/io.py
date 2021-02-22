import os
import requests
import tqdm
import typing
import yaml
import zipfile


def download_and_unzip(url: str, folder: str) -> str:
    """Download and unzip file at the given folder. Afterwards delete the zip file again.

    :param url: download file from this url.
    :param folder: write downloaded files to this path.
    :return: global path to unpacked file directory.
    """
    assert type(url) == str and "?" in url, f"Invalid url {url} for downloading"

    filename = url.rpartition('/')[2].split('?')[0]
    if not filename.endswith(".zip"):
        filename += ".zip"
    path = os.path.join(folder, filename)
    path_unpacked = path.replace(".zip", "")
    if os.path.exists(path_unpacked):
        return path_unpacked

    _ = download_file(url, path=path)
    with zipfile.ZipFile(path, 'r') as zip_obj:
        zip_obj.extractall(path=folder)
    os.remove(path)
    return path_unpacked


def download_file(url: str, path: str, chunk_size: int = 8192) -> str:
    """Download large files from the web.
    Adopted by: https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests

    :param url: download file from this url.
    :param path: write downloaded file to this path.
    :param chunk_size: number of bytes that are fetched at once.
    :return: global path of fetched file.
    """
    print(f"Downloading file to {path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        content_size = int(r.headers.get("Content-length", 0))
        num_chunks = content_size / chunk_size

        with open(path, 'wb') as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), total=num_chunks):
                f.write(chunk)
            f.close()
        r.close()

    return path


def read_yaml_to_dict(file_path: str, is_local: bool = True) -> typing.Dict[str, typing.Any]:
    """Read yaml file to dictionary object.

    :param file_path: path to yaml file that should be read.
    :param is_local: local or global file path.
    :return: contained dictionary.
    """
    if is_local:
        file_path = path_local_to_global(local_path=file_path)
    with open(file_path, "r") as file:
        return yaml.full_load(file)


def path_local_to_global(local_path: str) -> str:
    """Convert a local path, starting at the project's root directory, to a global path."""
    global_path = os.path.dirname(os.path.abspath(__file__))  # path of io.py file
    global_path = os.path.dirname(os.path.dirname(global_path))
    return os.path.join(global_path, local_path)
