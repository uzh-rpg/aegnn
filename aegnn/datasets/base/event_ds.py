import glob
import logging
import numpy as np
import os
import torch
import torch_geometric

from torch.utils.data import Subset
from torch_geometric.data import Batch, Data, Dataset
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple, Union

from aegnn.utils.bounding_box import crop_to_frame, is_bbox_zero
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
    # Pre-process raw files to enable more efficient loading during training/inference.
    # The process function is called by the `torch_geometric.data.Dataset` super-class when the processed
    # directory is empty, or when not all files that are listed as processed files can be found. This function
    # is made to process the whole dataset, not single files, in parallel processing.
    # Loading the data, annotations, labels, etc. is dataset specific and therefore to be defined in the
    # sub-classes. Importantly, the functions must therefore be able to be pickled, i.e. not using dynamic
    # types or not-shared class variables.
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
        kwargs = dict(raw_dir=self.raw_dir, target_dir=self.processed_dir, load_func=self.load,
                      pre_filter=self.pre_filter, pre_transform=self.pre_transform, img_shape=self.img_shape,
                      get_sections=self.get_sections, build_meta_info=self.build_meta_info,
                      read_label=self.read_label, read_annotations=self.read_annotations)
        if self.num_workers > 1:
            logging.info(f"Processing raw files with {self.num_workers} workers")
            task_manager = TaskManager(self.num_workers, queue_size=self.num_workers, total=len(self.raw_paths))
            for rf in self.raw_paths:
                task_manager.queue(self.processing, rf=rf, **kwargs)
            task_manager.join()
        else:
            logging.info("Processing raw files with a single process")
            for rf in tqdm(self.raw_paths):
                self.processing(rf, **kwargs)

    @staticmethod
    def processing(rf: str, raw_dir: str, target_dir: str,
                   load_func: Callable[[str], Data],
                   read_label: Callable[[str], str], read_annotations: Callable[[str], np.ndarray],
                   get_sections: Callable[[Data], List[Data]], img_shape: Tuple[int, int],
                   build_meta_info: Callable[[Data], Dict[str, str]],
                   pre_filter: Callable = None, pre_transform: Callable = None):
        rf_wo_ext, _ = os.path.splitext(rf)
        pf = rf_wo_ext.replace(raw_dir, target_dir) + ".pt"
        pf_rel = os.path.relpath(pf, start=target_dir)
        pf_flat = os.path.join(target_dir, pf_rel.replace("/", "."))

        # Load data from raw file. If the according loaders are available, add annotation, label and class id.
        device = torch.device(torch.cuda.current_device())
        data_obj = load_func(rf).to(device)
        data_obj.file_id = os.path.basename(rf)
        if (label := read_label(rf)) is not None:
            data_obj.label = label if isinstance(label, list) else [label]
        if (bbox := read_annotations(rf)) is not None:
            bbox = crop_to_frame(torch.tensor(bbox, device=device), image_shape=img_shape)
            is_not_empty = is_bbox_zero(bbox)
            data_obj.bbox = bbox[~is_not_empty, :].long()
            if hasattr(data_obj, "label"):
                data_obj.label = [lbl for lbl, ie in zip(data_obj.label, is_not_empty) if not ie]

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
    # Data Loading ##########################################################################################
    #########################################################################################################
    @staticmethod
    def build_meta_info(data: Data) -> Dict[str, str]:
        return {"label": getattr(data, "label", None)}

    def get_subset(self, **kwargs) -> Subset:
        """Get the dataset subset based on the meta information stored next to each data file
        (see `build_meta_info`). However, while efficient in data index filtering, the subset
        cannot change the data itself, just choose samples (-> `collate`).

        :param kwargs: key-value pairs for filtering (key = meta key, value = list of contained values).
        """
        logging.debug(f"Creating subset based on filters {kwargs}")
        assert all([isinstance(v, list) for v in kwargs.values()]), "values should be lists"
        indices = []

        for i, pf in enumerate(self.processed_paths):
            meta_dict = torch.load(pf.replace(".data.pt", ".meta"))
            is_inside = True

            # For each key-value pair in the filtering keys (`kwargs`), check whether the key has been
            # stored in the files meta information. Let the file through, if
            # a) the key is not stored in the meta file
            # b) the values are a subset of the filter values.
            for key, values in kwargs.items():
                p_values = meta_dict.get(key, None)
                if p_values is None:
                    continue
                elif not set(p_values).isdisjoint(set(values)):
                    continue
                is_inside = False
                break

            # If the file is inside the filter kwargs, add its index to the list of subset indices.
            if is_inside:
                indices.append(i)
        return Subset(self, indices=indices)

    def collate(self, batch: Batch) -> Batch:
        """Collate a batch by loading and filtering the data as well as the annotations (labels, bounding boxes, etc.).
        Thereby, only load the internally declared classes list and filter certain "not useful" annotations.

        :param batch: input batch to collate/filter.
        """
        batch_new = []
        label_dict = {label: i for i, label in enumerate(self.classes)}

        for batch_idx, data in enumerate(batch):
            # Filter the bounding boxes by the label (if e.g. only certain classes should be labeled) and
            # by the bounding box, which depends on the dataset.
            label_mask = [lbl in self.classes for lbl in data.label]
            bbox_mask = self.bbox_mask(data.bbox)
            arg_label = np.where(np.logical_and(bbox_mask, label_mask))[0].tolist()

            data.batch_bbox = torch.ones(len(arg_label)) * batch_idx
            data.label = [data.label[i] for i in arg_label]
            class_id = [label_dict[lbl] for lbl in data.label]
            with torch.no_grad():
                class_id = torch.tensor(class_id, device=data.bbox.device)
                data.bbox = data.bbox[arg_label, :]
                data.bbox[..., -1] = class_id
                data.y = class_id.long()

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
