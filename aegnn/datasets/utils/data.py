import os
import numpy as np
import torch
import torch_geometric

from tqdm import tqdm
from typing import Callable, List, Tuple

import aegnn.datasets.utils as utils


def process_events(load_func: Callable[[str], Tuple[torch_geometric.data.Data, str]], raw_dir: str,
                   raw_files: List[str], target_dir: str, classes: List[str],
                   annotations_dir: str = None, read_annotations: Callable[[str], np.ndarray] = None,
                   pre_filter=None, pre_transform=None):
    files_file = os.path.join(target_dir, "files.txt")

    object_class_ids = {class_id: i for i, class_id in enumerate(classes)}
    class_id = len(classes)
    processed_files = []

    cuda_id = torch.cuda.current_device()
    with torch.cuda.device(cuda_id):
        for rf in tqdm(raw_files):
            pf = rf.replace(raw_dir, target_dir).replace(".npy", ".pt")
            data_obj, obj_class = load_func(rf)

            # Add label attribute to data object, based on iterating over the unique class ids.
            if obj_class not in object_class_ids.keys():
                object_class_ids[obj_class] = class_id
                class_id += 1
            data_obj.y = object_class_ids[obj_class]

            # Apply pre-filter and pre-transform to the data object, if defined.
            if pre_filter is not None and not pre_filter(data_obj):
                continue
            if pre_transform is not None:
                data_obj = pre_transform(data_obj)

            # Add annotations, if they are defined for the given file.
            if annotations_dir:
                af = rf.replace(raw_dir, annotations_dir)
                data_obj.bb = read_annotations(af)

            os.makedirs(os.path.dirname(pf), exist_ok=True)
            torch.save(data_obj.to("cpu"), pf)
            processed_files.append(os.path.relpath(pf, target_dir))

    utils.io.write_list_to_file(processed_files, file_path=files_file)
    return files_file
