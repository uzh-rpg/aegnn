import glob
import os
import numpy as np
import scipy.io as sio
import torch
import torch_geometric

from typing import Tuple

import aegnn.datasets.utils as utils
from .base import EventDataset


class CIFAR10(EventDataset):

    class CIFAR10DS(torch_geometric.data.Dataset):

        def __init__(self, root: str, transform, pre_transform, pre_filter, classes=None, mode: str = "training"):
            root_mode = os.path.join(root, mode)
            if not os.path.isdir(root_mode):
                raise FileNotFoundError(f"Mode {mode} not found at root {root}!")

            self.classes = ["dog", "airplane", "bird"]
            super().__init__(root=root_mode, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        def get(self, idx: int) -> torch_geometric.data.Data:
            data_file = str(os.path.join(self.processed_dir, self.processed_file_names[idx]))
            return torch.load(data_file)

        #########################################################################################################
        # Processing ############################################################################################
        #########################################################################################################
        def process(self):
            def load(raw_file: str) -> Tuple[torch_geometric.data.Data, str]:
                saved_name = os.path.basename(raw_file).replace(".mat", ".pt")

                content = sio.loadmat(raw_file)
                obj_class = saved_name.split("_")[1]  # format: cifar10_dog_2.pt
                feature = torch.tensor(content['feature'])
                edge_index = torch.tensor(np.array(content['edge'], np.int32), dtype=torch.long)
                pos = torch.tensor(content['pseudo'])
                data_obj = torch_geometric.data.Data(x=feature, edge_index=edge_index, class_id=obj_class,
                                                     file_id=saved_name, pos=pos)
                return data_obj, obj_class

            utils.data.process_events(load, raw_dir=self.raw_dir, raw_files=self.raw_file_names,
                                      target_dir=self.processed_dir, classes=self.classes,
                                      pre_filter=self.pre_filter, pre_transform=self.pre_transform)

        #########################################################################################################
        # Files #################################################################################################
        #########################################################################################################
        @property
        def raw_file_names(self):
            return glob.glob(os.path.join(self.raw_dir, "*.mat"), recursive=True)

        @property
        def processed_file_names(self):
            return utils.io.read_txt_to_list(os.path.join(self.processed_dir, "files.txt"), default=["unknown"])

        def len(self) -> int:
            return len(self.processed_file_names)

    #########################################################################################################
    # Data Module ###########################################################################################
    #########################################################################################################
    def __init__(self, **kwargs):
        super().__init__(dataset_class=self.CIFAR10DS, **kwargs)

    @property
    def img_shape(self) -> Tuple[int, int]:
        return 128, 128
