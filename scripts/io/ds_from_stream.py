import argparse
import logging
import h5py
import numpy as np
import glob
import os
import torch

from torch_geometric.data import Data
from tqdm import tqdm
from typing import Tuple
from aegnn.utils import TaskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size-bd", action="store", default=90000, type=int,
                        help="Number of events in window, going backward in time")
    parser.add_argument("--window-size-fd", action="store", default=10000, type=int,
                        help="Number of events in window, going backward in time")
    parser.add_argument("--num-workers", action="store", default=20, type=int,
                        help="Number of workers for parallel processing")
    return parser.parse_args()


#########################################################################################################
# Reading Data ##########################################################################################
#########################################################################################################
def read_data(f_path: str):
    f = h5py.File(f_path, 'r')
    x = f["events"].get("x")
    y = f["events"].get("y")
    t = f["events"].get("t")
    p = f["events"].get("p")
    return x, y, t, p


def read_annotation(h5_file: str) -> Tuple[np.ndarray, np.ndarray]:
    annotation_file = h5_file.replace("_td.h5", "_box.npy")
    bbox = np.load(annotation_file)
    return np.array([bbox["x"], bbox["y"], bbox["w"], bbox["h"], bbox["class_id"]]).T, bbox["t"]


def read_bbox_to_id(h5_file: str) -> np.ndarray:
    file_name = os.path.basename(h5_file).replace("_td.h5", "_bbox_to_event_idx.npy")
    ds = os.path.basename(os.path.dirname(h5_file))
    directory = os.path.dirname(os.path.dirname(h5_file))
    f_path = os.path.join(directory, "bbox_to_idx", ds, file_name)
    return np.load(f_path)


#########################################################################################################
# Processing video ######################################################################################
#########################################################################################################
def process(f_path: str, ds: str, window_size_bd: int, window_size_fd: int, **kwargs):
    file_name = os.path.basename(f_path).replace("_td.h5", "")
    x, y, t, p = read_data(f_path)
    bbox, t_bbox = read_annotation(f_path)
    bbox_to_idx = read_bbox_to_id(f_path)

    num_events = x.size
    for bbox_idx, idx in tqdm(bbox_to_idx):
        ws = max(idx - window_size_bd, 0)
        we = min(idx + window_size_fd, num_events - 1)

        w_bbox_mask = np.logical_and(t_bbox >= t[ws], t_bbox <= t[we])
        w_bbox = bbox[w_bbox_mask, :]
        w_pos = np.stack([x[ws:we], y[ws:we], t[ws:we] - t[ws]], axis=-1).astype(np.uint32)
        w_p = p[ws:we].astype(bool)

        output_directory = os.path.join(os.environ["AEGNN_DATA_DIR"], "megapixel", ds, "raw")
        os.makedirs(output_directory, exist_ok=True)
        pf = os.path.join(output_directory, file_name + f".{bbox_idx}.pt")

        data = Data(x=w_p, y=w_bbox[:, -1], pos=w_pos, bbox=w_bbox)
        torch.save(data, pf)


#########################################################################################################
# __main__ ##############################################################################################
#########################################################################################################
if __name__ == '__main__':
    args = parse_args()
    dataset_dir = os.environ["AEGNN_DATASET_DIR"]

    for ds_type in ["train", "val"]:
        logging.info(f"Processing {ds_type} dataset")
        dataset_dir = os.path.join(dataset_dir, f"oneMP_add/Large_Automotive_Detection_Dataset_extracted/{ds_type}/")
        h5_files = glob.glob(os.path.join(dataset_dir, "*"))
        h5_files = list(filter(lambda f: f.endswith(".h5"), h5_files))

        task_manager = TaskManager(args.num_workers, queue_size=args.num_workers, total=len(h5_files))
        for i, h5file in enumerate(h5_files):
            if i > 10:
                break
            task_manager.queue(process, f_path=h5file, ds=ds_type, **vars(args))
