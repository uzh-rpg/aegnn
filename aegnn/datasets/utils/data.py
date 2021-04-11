import os
import numpy as np
import random
import torch
import torch_geometric

from tqdm import tqdm
from typing import Callable, Dict, List

import aegnn.datasets.utils as utils


def process_events(load_func: Callable[[str], torch_geometric.data.Data], raw_dir: str,
                   raw_files: List[str], target_dir: str, classes: List[str], read_class_id: Callable[[str], str],
                   annotations_dir: str = None, read_annotations: Callable[[str], np.ndarray] = None,
                   pre_filter=None, pre_transform=None, wdt: float = 0.03):
    print("Building class dictionary")
    object_class_ids = __build_class_dict(raw_files, classes=classes, read_class_id=read_class_id)

    print("Processing raw files")
    files_file = os.path.join(target_dir, "files.txt")
    cuda_id = torch.cuda.current_device()
    with torch.cuda.device(cuda_id):
        processed_files = []
        for rf in tqdm(raw_files):
            pf = __make_p_file(rf, raw_dir=raw_dir, target_dir=target_dir)
            data_obj = load_func(rf)

            # Add label attribute to data object, based on iterating over the unique class ids.
            if (class_id := getattr(data_obj, "class_id", None)) is not None:
                data_obj.y = object_class_ids[class_id]

            # Apply pre-filter and pre-transform to the data object, if defined.
            if pre_filter is not None and not pre_filter(data_obj):
                return []

            # Add annotations, if they are defined for the given file.
            if annotations_dir:
                af = rf.replace(raw_dir, annotations_dir)
                data_obj.bb = read_annotations(af)

            # Filter a random window of length `dt` from the sample. To do so, find the number of
            # windows with length `dt` in the data, sample one of them and filter the data accordingly.
            assert data_obj.pos.size(1) > 2
            for t_section in range(3):
                do_section = __get_t_section(data_obj, wdt=wdt)
                if do_section.num_nodes == 0:
                    continue

                if pre_transform is not None:
                    do_section = pre_transform(do_section)

                # Save the section data object as .pt-torch-file and append the filename to the
                # list of pre-processed files.
                pf_section = pf.replace(".pt", f".{t_section}.pt")
                os.makedirs(os.path.dirname(pf_section), exist_ok=True)
                torch.save(do_section.to("cpu"), pf_section)
                processed_files.append(os.path.relpath(pf_section, start=target_dir))

    utils.io.write_list_to_file(processed_files, file_path=files_file)
    return files_file


def __make_p_file(raw_file: str, raw_dir: str, target_dir: str) -> str:
    rf_wo_ext, _ = os.path.splitext(raw_file)
    return rf_wo_ext.replace(raw_dir, target_dir) + ".pt"

def __build_class_dict(raw_files: List[str], classes: List[str], read_class_id: Callable[[str], str]) -> Dict[str, int]:
    object_class_ids = {class_id: i for i, class_id in enumerate(classes)}
    class_id = len(classes)

    for rf in tqdm(raw_files):
        obj_class = read_class_id(rf)
        if obj_class not in object_class_ids.keys():
            object_class_ids[obj_class] = class_id
            class_id += 1
    return object_class_ids


def __get_t_section(data_obj: torch_geometric.data.Data, wdt: float) -> torch_geometric.data.Data:
    t_min = data_obj.pos[:, 2].min()
    t_max = data_obj.pos[:, 2].max()
    data = data_obj.clone()

    t_start = t_min + random.random() * (t_max - t_min - wdt)
    t_end = t_start + wdt
    idx_select = torch.logical_and(t_start <= data.pos[:, 2], data.pos[:, 2] < t_end)
    data.x = data.x[idx_select, :]
    data.pos = data.pos[idx_select, :]
    return data
