import argparse
import collections
import glob
import os
import random
import shutil

from tqdm import tqdm
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", action="store")
    parser.add_argument("--directory", action="store", default="raw")
    parser.add_argument("--test-split", action="store", type=float, default=0.2)
    parser.add_argument("--root", action="store", type=str, default="/data/storage/simonschaefer")
    parser.add_argument("--seed", action="store", type=int, default=12345)
    return parser.parse_args()


def copy_files_to_directory(files: List[str], source_dir: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    for f_source in files:
        f_target = f_source.replace(source_dir, target_dir)
        os.makedirs(os.path.dirname(f_target), exist_ok=True)
        shutil.copyfile(f_source, f_target)



if __name__ == '__main__':
    args = parse_args()
    if args.test_split < 0:
        raise ValueError(f"Invalid testing split {args.test_split} < 0!")
    random.seed(args.seed)

    root_path = os.path.join(args.root, args.dataset)
    train_path = os.path.join(root_path, "training", "raw")
    val_path = os.path.join(root_path, "validation", "raw")

    # Iterate over all classes, split them in training and testing samples and copy the
    # assigned files to either directory.
    raw_path = os.path.join(root_path, args.directory)
    for class_id in tqdm(os.listdir(raw_path)):
        class_path = os.path.join(raw_path, class_id)
        class_files = glob.glob(f"{class_path}/*")

        val_files = random.sample(class_files, k=int(len(class_files) * args.test_split))  # w/o replacement
        copy_files_to_directory(val_files, source_dir=raw_path, target_dir=val_path)
        train_files = [f for f in class_files if f not in val_files]
        copy_files_to_directory(train_files, source_dir=raw_path, target_dir=train_path)
