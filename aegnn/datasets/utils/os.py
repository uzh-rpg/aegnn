import os


def path_local_to_global(local_path: str) -> str:
    """Convert a local path, starting at the project's root directory, to a global path."""
    global_path = os.path.dirname(os.path.abspath(__file__))  # path of io.py file
    global_path = os.path.dirname(os.path.dirname(global_path))
    return os.path.join(global_path, local_path)
