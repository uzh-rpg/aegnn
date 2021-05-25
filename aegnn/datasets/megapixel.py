import functools
import glob
import os
import numpy as np
import torch

from torch_geometric.data import Data
from typing import List, Tuple, Union

from .base.event_ds import EventDataset
from .base.event_dm import EventDataModule


class Megapixel(EventDataModule):

    class MegapixelDS(EventDataset):

        def process(self):
            """Required by `torch_geometric.data.Dataset` in order to start processing."""
            super().process()

        def read_annotations(self, raw_file: str) -> np.ndarray:
            bbox = self._load_file(raw_file).bbox
            is_pedestrian = bbox[:, -1] == 0
            is_car = np.logical_or(np.logical_or(bbox[:, -1] == 2, bbox[:, -1] == 3), bbox[:, -1] == 4)
            bbox = bbox[np.logical_or(is_pedestrian, is_car), :]

            bbox[is_pedestrian, -1] = 0
            bbox[is_car, -1] = 1
            return bbox

        def read_label(self, raw_file: str) -> Union[str, List[str], None]:
            class_id = self.read_annotations(raw_file)[:, -1].tolist()
            label_dict = {0: "pedestrian", 1: "car"}
            return [label_dict.get(cid, None) for cid in class_id]

        def load(self, raw_file: str) -> Data:
            data = self._load_file(raw_file)
            x = torch.from_numpy(data.x).float()
            pos = torch.from_numpy(data.pos).float()
            return Data(x=x, pos=pos)

        @functools.lru_cache(maxsize=10)
        def _load_file(self, f_path: str):
            return torch.load(f_path)

        #########################################################################################################
        # Files #################################################################################################
        #########################################################################################################
        @property
        def raw_files(self):
            return glob.glob(os.path.join(self.raw_dir, "*.pt"))

        @property
        def raw_classes(self) -> List[str]:
            return ["pedestrian", "car"]

        #########################################################################################################
        # Meta ##################################################################################################
        #########################################################################################################
        @property
        def img_shape(self) -> Tuple[int, int]:
            return 1280, 720

    #########################################################################################################
    # Data Module ###########################################################################################
    #########################################################################################################
    def __init__(self, **kwargs):
        super().__init__(dataset_class=self.MegapixelDS, **kwargs)
