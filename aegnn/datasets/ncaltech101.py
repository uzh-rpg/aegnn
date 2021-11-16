import functools
import glob
import logging
import numpy as np
import os
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union

from aegnn.utils.multiprocessing import TaskManager
from .base.event_dm import EventDataModule
from .utils.normalization import normalize_time


class NCaltech101(EventDataModule):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8,
                 pin_memory: bool = False, transform: Optional[Callable[[Data], Data]] = None):
        super(NCaltech101, self).__init__(img_shape=(240, 180), batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers, pin_memory=pin_memory, transform=transform)
        pre_processing_params = {"r": 5.0, "d_max": 32, "n_samples": 25000, "sampling": True}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        annotations_dir = os.path.join(os.environ["AEGNN_DATA_DIR"], "ncaltech101", "annotations")
        raw_file_name = os.path.basename(raw_file).replace("image", "annotation")
        raw_dir_name = os.path.basename(os.path.dirname(raw_file))
        annotation_file = os.path.join(os.path.join(annotations_dir, raw_dir_name, raw_file_name))

        f = open(annotation_file)
        annotations = np.fromfile(f, dtype=np.int16)
        annotations = np.array(annotations[2:10])
        f.close()

        label = self.read_label(raw_file)
        class_id = self.map_label(label)
        if class_id is None:
            return None

        # Create bounding box from corner, shape and label variables. NCaltech101 bounding boxes
        # often start outside of the frame (negative corner coordinates). However, the shape turns
        # out to be the shape of the bbox starting at the image's frame.
        bbox = np.array([
            annotations[0], annotations[1],  # upper-left corner
            annotations[2] - annotations[0],  # width
            annotations[5] - annotations[1],  # height
            class_id
        ])
        bbox[:2] = np.maximum(bbox[:2], 0)
        return bbox.reshape((1, 1, -1))

    @staticmethod
    def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
        return raw_file.split("/")[-2]

    @staticmethod
    def load(raw_file: str) -> Data:
        f = open(raw_file, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()

        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        all_ts = all_ts / 1e6  # µs -> s
        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1
        events = np.column_stack((all_x, all_y, all_ts, all_p))
        events = torch.from_numpy(events).float().cuda()

        x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        return Data(x=x, pos=pos)

    @functools.lru_cache(maxsize=100)
    def map_label(self, label: str) -> int:
        label_dict = {lbl: i for i, lbl in enumerate(self.classes)}
        return label_dict.get(label, None)

    def _load_processed_file(self, f_path: str) -> Data:
        return torch.load(f_path)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    def _prepare_dataset(self, mode: str):
        processed_dir = os.path.join(self.root, "processed")
        raw_files = self.raw_files(mode)
        class_dict = {class_id: i for i, class_id in enumerate(self.classes)}
        kwargs = dict(load_func=self.load, class_dict=class_dict, pre_transform=self.pre_transform,
                      read_label=self.read_label, read_annotations=self.read_annotations)
        logging.debug(f"Found {len(raw_files)} raw files in dataset (mode = {mode})")

        task_manager = TaskManager(self.num_workers, queue_size=self.num_workers)
        processed_files = []
        for rf in tqdm(raw_files):
            processed_file = rf.replace(self.root, processed_dir)
            processed_files.append(processed_file)

            if os.path.exists(processed_file):
                continue
            task_manager.queue(self.processing, rf=rf, pf=processed_file, **kwargs)
        task_manager.join()

    @staticmethod
    def processing(rf: str, pf: str, load_func: Callable[[str], Data],
                   class_dict: Dict[str, int], read_label: Callable[[str], str],
                   read_annotations: Callable[[str], np.ndarray], pre_transform: Callable = None):
        rf_wo_ext, _ = os.path.splitext(rf)

        # Load data from raw file. If the according loaders are available, add annotation, label and class id.
        device = "cpu"  # torch.device(torch.cuda.current_device())
        data_obj = load_func(rf).to(device)
        data_obj.file_id = os.path.basename(rf)
        if (label := read_label(rf)) is not None:
            data_obj.label = label if isinstance(label, list) else [label]
            data_obj.y = torch.tensor([class_dict[label] for label in data_obj.label])
        if (bbox := read_annotations(rf)) is not None:
            data_obj.bbox = torch.tensor(bbox, device=device).long()

        # Apply the pre-transform on the graph, to afterwards store it as .pt-file.
        assert data_obj.pos.size(1) == 3, "pos must consist of (x, y, t)"
        if pre_transform is not None:
            data_obj = pre_transform(data_obj)

        # Save the data object as .pt-torch-file. For the sake of a uniform processed
        # directory format make all output paths flat.
        os.makedirs(os.path.dirname(pf), exist_ok=True)
        torch.save(data_obj.to("cpu"), pf)

    def pre_transform(self, data: Data) -> Data:
        params = self.hparams.preprocessing

        # Cut-off window of highest increase of events.
        window_us = 50 * 1000
        t = data.pos[data.num_nodes // 2, 2]
        index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
        index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-window_us) - 1, 0, data.num_nodes - 1)
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == data.num_nodes and item.size(0) != 1:
                data[key] = item[index0:index1, :]

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = normalize_time(data.pos[:, 2])
        # Radius graph generation.
        data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
        return data

    @staticmethod
    def sub_sampling(data: Data, n_samples: int, sub_sample: bool) -> Data:
        if sub_sample:
            sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
            return sampler(data)
        else:
            sample_idx = np.arange(n_samples)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) != 1:
                    data[key] = item[sample_idx]
            return data

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode, "*", "*.bin"), recursive=True)

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode, "*", "*.bin"))

    @property
    def classes(self) -> List[str]:
        return os.listdir(os.path.join(self.root, "raw"))
