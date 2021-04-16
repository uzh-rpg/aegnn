import argparse
import collections
import glob
import os
import shutil

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", action="store")
    parser.add_argument("--directories", action="store", type=lambda x: x.split(":"), default=[])
    parser.add_argument("--root", action="store", type=str, default="/data/storage/simonschaefer")
    parser.add_argument("--out", action="store", default="out", type=str)
    parser.add_argument("--suffix", action="store", default="npy", type=str)
    return parser.parse_args()


def get_class(file_name: str) -> str:
    return file_name.split("/")[-2]

if __name__ == '__main__':
    args = parse_args()
    if len(args.directories) < 1:
        raise ValueError(f"Invalid number of input datasets: {args.directories}")
    root_path = os.path.join(args.root, args.dataset)

    # List all files over the input directories.
    files = []
    class_file_dict = collections.defaultdict(list)
    for directory in args.directories:
        dir_path = os.path.join(root_path, directory)
        files_in_directory = glob.glob(os.path.join(dir_path, "*", "*", f"*.{args.suffix}"))
        for file in files_in_directory:
            file_class = get_class(file_name=file)
            class_file_dict[file_class].append(file)

    # Copy (!) all files from their directories to the output directory. In order to avoid
    # naming conflicts, re-number them.
    out_path = os.path.join(root_path, args.out)
    os.makedirs(out_path, exist_ok=True)
    for class_id, files in tqdm(class_file_dict.items()):
        os.makedirs(os.path.join(out_path, "raw", class_id), exist_ok=True)
        for i, file in enumerate(files):
            file_new = os.path.join(out_path, "raw", class_id, f"{class_id}_{i+1}.{args.suffix}")
            shutil.copyfile(file, file_new)
