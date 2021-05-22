import glob
import logging
import numpy as np
import os
import torch
import torch_geometric

from typing import Callable, List, Union

from aegnn.utils.multiprocessing import TaskManager


class EventDataset(torch_geometric.data.Dataset):

    def __init__(self, root: str, transforms: Callable, pre_transform: Callable, pre_filter: Callable,
                 classes: List[str] = None, mode: str = "training", num_workers: int = 1):
        root_mode = os.path.join(root, mode)
        if not os.path.isdir(root_mode):
            raise FileNotFoundError(f"Mode {mode} not found at root {root}!")

        self.classes = classes or os.listdir(os.path.join(root_mode, "raw"))
        self.num_workers = num_workers
        transform = torch_geometric.transforms.Compose(transforms)
        super().__init__(root=root_mode, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def get(self, idx: int) -> torch_geometric.data.Data:
        data_file = str(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return torch.load(data_file)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    def read_annotations(self, raw_file: str) -> Union[np.ndarray, None]:
        return None

    def read_class_id(self, raw_file: str) -> Union[int, None]:
        return None

    def read_label(self, raw_file: str) -> Union[str, None]:
        return None

    def load(self, raw_file: str) -> torch_geometric.data.Data:
        raise NotImplementedError

    def get_sections(self, data_obj: torch_geometric.data.Data) -> List[torch_geometric.data.Data]:
        return [data_obj]

    def process(self):
        # Processing the raw files in parallel processes. Importantly, the functions must therefore be able to be
        # pickled, i.e. not using dynamic types or not-shared class variables.
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
                    get_sections=self.get_sections, read_label=self.read_label,
                    read_annotations=self.read_annotations, read_class_id=self.read_class_id)

    @staticmethod
    def processing(rf: str, raw_dir: str, target_dir: str,
                   load_func: Callable[[str], torch_geometric.data.Data], read_label: Callable[[str], str],
                   read_class_id: Callable[[str], int], read_annotations: Callable[[str], np.ndarray],
                   get_sections: Callable[[torch_geometric.data.Data], List[torch_geometric.data.Data]],
                   pre_filter: Callable = None, pre_transform: Callable = None):
        """Processing raw file on cuda device for object detection/recognition task, including the following steps:

        1. Loading and converting the data to `torch_geometric.data.Data` object.
        2. Assigning y based on class id
        3. Filtering the data out, if required.
        4. Add annotations such as bounding boxes.
        5. Crop several selected windows from the data.
        6. Perform the pre-processing transformation on each selected window.
        7. Save the resulting objects in the `target_dir` as torch file.

        :param rf: absolute path to raw file to process.
        :param raw_dir: absolute path to the directory of raw files.
        :param target_dir: absolute path to the directory to save processed data.
        :param load_func: function to load raw file as data object (str -> `torch_geometric.data.Data`).
        :param read_label: function to read the label given the raw data file (str -> str).
        :param read_class_id: function to read the class id given the raw data file (str -> int).
        :param read_annotations: function to read annotations given the raw data file (str -> `np.array`).
        :param get_sections: divide the full loaded data into sections (`torch_geometric.data.Data` -> List[Data]).
        :param pre_filter: pre-filtering data object (`torch_geometric.data.Data` -> bool).
        :param pre_transform: transforming data object (`torch_geometric.data.Data` -> `torch_geometric.data.Data`).
        """
        cuda_id = torch.cuda.current_device()

        with torch.cuda.device(cuda_id):
            rf_wo_ext, _ = os.path.splitext(rf)
            pf = rf_wo_ext.replace(raw_dir, target_dir) + ".pt"
            pf_rel = os.path.relpath(pf, start=target_dir)
            pf_flat = os.path.join(target_dir, pf_rel.replace("/", "."))

            # Load data from raw file. If the according loaders are available, add annotation, label and class id.
            data_obj = load_func(rf)
            if (label := read_label(rf)) is not None:
                data_obj.label = list(label)
            if (class_id := read_class_id(rf)) is not None:
                data_obj.class_id = int(class_id)
            if (bbox := read_annotations(rf)) is not None:
                data_obj.bbox = torch.tensor(bbox)

            # Apply pre-filter and pre-transform to the data object, if defined.
            if pre_filter is not None and not pre_filter(data_obj):
                return []

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
                torch.save(section.to("cpu"), pf_flat.replace(".pt", f".{i}.pt"))

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    @property
    def raw_files(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        return [os.path.relpath(f, start=self.raw_dir) for f in self.raw_files]

    @property
    def processed_file_names(self):
        files = glob.glob(os.path.join(self.processed_dir, "*.pt"), recursive=True)
        if len(files) == 0:
            return ["unknown"]
        return [os.path.relpath(f, start=self.processed_dir) for f in files]

    def len(self) -> int:
        return len(self.processed_file_names)
