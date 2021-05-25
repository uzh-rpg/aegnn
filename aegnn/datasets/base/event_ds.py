import glob
import logging
import numpy as np
import os
import torch
import torch_geometric

from torch.utils.data import Subset
from torch_geometric.data import Batch, Data, Dataset
from typing import Callable, Dict, List, Tuple, Union

from aegnn.utils.multiprocessing import TaskManager


class EventDataset(Dataset):

    def __init__(self, root: str, transforms: Callable, pre_transform: Callable, pre_filter: Callable,
                 classes: List[str] = None, mode: str = "training", num_workers: int = 1):
        root_mode = os.path.join(root, mode)
        if not os.path.isdir(root_mode):
            raise FileNotFoundError(f"Mode {mode} not found at root {root}!")
        transform = torch_geometric.transforms.Compose(transforms)
        self.num_workers = num_workers
        self.root = root_mode
        self.classes = classes or self.raw_classes
        super().__init__(root=root_mode, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def get(self, idx: int) -> Data:
        data_file = str(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return torch.load(data_file)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    def read_annotations(self, raw_file: str) -> Union[np.ndarray, None]:
        return None

    def read_label(self, raw_file: str) -> Union[str, List[str], None]:
        return None

    def load(self, raw_file: str) -> Data:
        raise NotImplementedError

    def get_sections(self, data_obj: Data) -> List[Data]:
        return [data_obj]

    def process(self):
        """Pre-process raw files to enable more efficient loading during training/inference.

        The process function is called by the `torch_geometric.data.Dataset` super-class when the processed
        directory is empty, or when not all files that are listed as processed files can be found. This function
        is made to process the whole dataset, not single files, in parallel processing.

        Loading the data, annotations, labels, etc. is dataset specific and therefore to be defined in the
        sub-classes. Importantly, the functions must therefore be able to be pickled, i.e. not using dynamic
        types or not-shared class variables.
        """
        if self.num_workers > 1:
            logging.info(f"Processing raw files with {self.num_workers} workers")
            task_manager = TaskManager(self.num_workers, queue_size=self.num_workers, total=len(self.raw_paths))
            process, args = task_manager.queue, [self.processing]
        else:
            logging.info("Processing raw files with a single process")
            process, args = self.processing, []

        # Processing raw files either in parallel or sequentially (debug).
        for rf in self.raw_paths:
            process(*args, rf=rf, raw_dir=self.raw_dir, target_dir=self.processed_dir, load_func=self.load,
                    pre_filter=self.pre_filter, pre_transform=self.pre_transform,
                    get_sections=self.get_sections, build_meta_info=self.build_meta_info,
                    read_label=self.read_label, read_annotations=self.read_annotations)

    @staticmethod
    def processing(rf: str, raw_dir: str, target_dir: str,
                   load_func: Callable[[str], Data],
                   read_label: Callable[[str], str], read_annotations: Callable[[str], np.ndarray],
                   get_sections: Callable[[Data], List[Data]],
                   build_meta_info: Callable[[Data], Dict[str, str]],
                   pre_filter: Callable = None, pre_transform: Callable = None):
        """Processing raw file on cuda device for object detection/recognition task, including the following steps:

        1. Loading and converting the data to `Data` object.
        2. Load additional data such as class_id, bounding box, etc.
        3. Filtering the data out, if required.
        4. Crop several selected windows from the data.
        5. Perform the pre-processing transformation on each selected window.
        6. Save the resulting objects in the `target_dir` as torch file.
        7. Save meta information about the data object for quick access."""
        rf_wo_ext, _ = os.path.splitext(rf)
        pf = rf_wo_ext.replace(raw_dir, target_dir) + ".pt"
        pf_rel = os.path.relpath(pf, start=target_dir)
        pf_flat = os.path.join(target_dir, pf_rel.replace("/", "."))

        # Load data from raw file. If the according loaders are available, add annotation, label and class id.
        device = torch.cuda.current_device()
        data_obj = load_func(rf).to(device)
        data_obj.file_id = os.path.basename(rf)
        if (label := read_label(rf)) is not None:
            data_obj.label = label if isinstance(label, list) else [label]
        if (bbox := read_annotations(rf)) is not None:
            data_obj.bbox = torch.tensor(bbox)

        # Apply pre-filter and pre-transform to the data object, if defined.
        if pre_filter is not None and not pre_filter(data_obj):
            return

        # Divide the loaded data into sections. Then for each section, apply the pre-transform
        # on the graph, to afterwards store it as .pt-file.
        assert data_obj.pos.size(1) > 2
        do_sections = get_sections(data_obj)
        for i, section in enumerate(do_sections):
            if section.num_nodes == 0:
                return
            if pre_transform is not None:
                section = pre_transform(section)

            # Save the section data object as .pt-torch-file. For the sake of a uniform processed
            # directory format make all output paths flat.
            os.makedirs(os.path.dirname(pf_flat), exist_ok=True)
            torch.save(section.to("cpu"), pf_flat.replace(".pt", f".{i}.data.pt"))

            # For building class-dependent subsets of data, write meta-information about the
            # data point in an additional text file, so that they can be accessed without having to
            # load the whole file.
            meta_info = build_meta_info(section)
            torch.save(meta_info, pf_flat.replace(".pt", f".{i}.meta"))

    #########################################################################################################
    # Meta-Information & Subset #############################################################################
    #########################################################################################################
    @staticmethod
    def build_meta_info(data: Data) -> Dict[str, str]:
        return {"label": getattr(data, "label", None)}

    def get_subset(self, **kwargs) -> Subset:
        logging.info(f"Creating subset based on filters {kwargs}")
        assert all([isinstance(v, list) for v in kwargs.values()]), "values should be lists"
        indices = []

        for i, pf in enumerate(self.processed_paths):
            meta_dict = torch.load(pf.replace(".data.pt", ".meta"))
            is_inside = True

            for key, values in kwargs.items():
                p_values = meta_dict.get(key, None)
                if p_values is None:
                    continue
                elif not set(p_values).isdisjoint(set(values)):
                    continue
                is_inside = False
                break

            if is_inside:
                indices.append(i)
        return Subset(self, indices=indices)

    def collate(self, batch: Batch) -> Batch:
        batch_new = []
        label_dict = {label: i for i, label in enumerate(self.classes)}

        for data in batch:
            arg_label = [i for i, lbl in enumerate(data.label) if lbl in self.classes]
            data.label = [data.label[i] for i in arg_label]
            class_id = [label_dict[lbl] for lbl in data.label]
            with torch.no_grad():
                class_id = torch.tensor(class_id, dtype=torch.long, device=data.bbox.device)
                data.bbox = data.bbox[arg_label, :]
                data.bbox[..., -1] = class_id
                data.y = class_id

            batch_new.append(data)
        return Batch.from_data_list(batch_new)

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    @property
    def raw_classes(self) -> List[str]:
        raise NotImplementedError

    @property
    def raw_files(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        return [os.path.relpath(f, start=self.raw_dir) for f in self.raw_files]

    @property
    def processed_file_names(self):
        files = glob.glob(os.path.join(self.processed_dir, "*.data.pt"), recursive=True)
        if len(files) == 0:
            return ["unknown"]
        return [os.path.relpath(f, start=self.processed_dir) for f in files]

    #########################################################################################################
    # Meta ##################################################################################################
    #########################################################################################################
    @property
    def img_shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def len(self) -> int:
        return len(self.processed_file_names)
