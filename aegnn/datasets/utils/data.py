import os
import random
import torch
import torch_geometric

from tqdm import tqdm
from typing import Callable, Dict, List



def processing(rf: str, raw_dir: str, target_dir: str, object_class_ids: Dict[str, int], wdt: float,
               load_func: Callable, pre_filter: Callable = None, pre_transform: Callable = None,
               annotations_dir: str = None, read_annotations: Callable = None):
    """Processing raw file on cuda device for object detection/recognition task, including the following steps:

    1. Loading and converting the data to `torch_geometric.data.Data` object.
    2. Assigning y based on class id
    3. Filtering the data out, if required.
    4. Add annotations such as bounding boxes.
    5. Crop several randomly selected windows from the data.
    6. Perform the pre-processing transformation on each selected window.
    7. Save the resulting objects in the `target_dir` as torch file.

    :param rf: absolute path to raw file to process.
    :param raw_dir: absolute path to the directory of raw files.
    :param target_dir: absolute path to the directory to save processed data.
    :param object_class_ids: map from class-ids to class label y.
    :param wdt: length of time-frame to select [s].
    :param load_func: function to load raw file as data object (str -> `torch_geometric.data.Data`).
    :param pre_filter: pre-filtering data object (`torch_geometric.data.Data` -> bool).
    :param pre_transform: transforming data object (`torch_geometric.data.Data` -> `torch_geometric.data.Data`).
    :param annotations_dir: absolute path to the directory of annotations.
    :param read_annotations: function to read annotation file (str -> `np.array`).
    """
    cuda_id = torch.cuda.current_device()
    out_files = []

    with torch.cuda.device(cuda_id):
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
            out_files.append(os.path.relpath(pf_section, start=target_dir))

    return out_files


def build_class_dict(raw_files: List[str], classes: List[str], read_class_id: Callable[[str], str]) -> Dict[str, int]:
    """Building a map from class_id to class label y by iterating over raw files, read their class-id and
    sequentially build up the dictionary.

    :param raw_files: absolute paths to raw files to read.
    :param classes: subset of classes that should be at the beginning of the dictionary.
    :param read_class_id: function to read the class id from the absolute raw file path (str -> str).
    """
    object_class_ids = {class_id: i for i, class_id in enumerate(classes)}
    class_id = len(classes)

    for rf in tqdm(raw_files):
        obj_class = read_class_id(rf)
        if obj_class not in object_class_ids.keys():
            object_class_ids[obj_class] = class_id
            class_id += 1
    return object_class_ids


#########################################################################################################
# File-Private utility functions ########################################################################
#########################################################################################################
def __make_p_file(raw_file: str, raw_dir: str, target_dir: str) -> str:
    rf_wo_ext, _ = os.path.splitext(raw_file)
    return rf_wo_ext.replace(raw_dir, target_dir) + ".pt"


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
