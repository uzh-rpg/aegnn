import glob
import numpy as np
import os
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, List, Optional, Union

from .utils.normalization import normalize_time
from .ncaltech101 import NCaltech101


class NCars(NCaltech101):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8, pin_memory: bool = False,
                 transform: Optional[Callable[[Data], Data]] = None):
        super(NCars, self).__init__(batch_size, shuffle, num_workers, pin_memory=pin_memory, transform=transform)
        self.dims = (120, 100)  # overwrite image shape
        pre_processing_params = {"r": 3.0, "d_max": 32, "n_samples": 10000, "sampling": True}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
        label_file = os.path.join(raw_file, "is_car.txt")
        with open(label_file, "r") as f:
            label_txt = f.read().replace(" ", "").replace("\n", "")
        return "car" if label_txt == "1" else "background"

    @staticmethod
    def load(raw_file: str) -> Data:
        events_file = os.path.join(raw_file, "events.txt")
        events = torch.from_numpy(np.loadtxt(events_file)).float().cuda()
        x, pos = events[:, -1:], events[:, :3]
        return Data(x=x, pos=pos)

    def pre_transform(self, data: Data) -> Data:
        params = self.hparams.preprocessing

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = normalize_time(data.pos[:, 2])

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])

        # Radius graph generation.
        data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
        return data

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode, "*"))

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode, "*"))

    @property
    def classes(self) -> List[str]:
        return ["car", "background"]
