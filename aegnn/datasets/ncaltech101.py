import glob
import numpy as np
import os
import torch
import torch_geometric

from typing import Tuple

import aegnn.datasets.utils as utils
from .base.event_ds import EventDataset


class NCaltech101(EventDataset):

    class NCaltech101DS(torch_geometric.data.Dataset):

        def __init__(self, root: str, transform, pre_transform, pre_filter, classes=None, mode: str = "training"):
            root_mode = os.path.join(root, mode)
            if not os.path.isdir(root_mode):
                raise FileNotFoundError(f"Mode {mode} not found at root {root}!")

            self.classes = classes or os.listdir(os.path.join(root_mode, "raw"))
            super().__init__(root=root_mode, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        def get(self, idx: int) -> torch_geometric.data.Data:
            data_file = str(os.path.join(self.processed_dir, self.processed_file_names[idx]))
            return torch.load(data_file)

        #########################################################################################################
        # Processing ############################################################################################
        #########################################################################################################
        def process(self):
            def load(raw_file: str) -> Tuple[torch_geometric.data.Data, str]:
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
                events = torch.from_numpy(events).cuda()

                obj_class = raw_file.split("/")[-2]
                file_id = raw_file.split("/")[-1]
                data_obj = torch_geometric.data.Data(x=events, pos=events[:, :3], class_id=obj_class, file_id=file_id)
                return data_obj, obj_class

            def read_annotations(annotation_file: str) -> np.ndarray:
                file_name = os.path.basename(annotation_file).replace("image", "annotation")
                f = open(os.path.join(os.path.dirname(annotation_file), file_name))
                annotations = np.fromfile(f, dtype=np.int16)
                f.close()
                return np.array(annotations[2:10])

            annotations_dir = os.path.join(os.environ["AEGNN_DATA_DIR"], "ncaltech101", "annotations")
            utils.data.process_events(load, raw_dir=self.raw_dir, raw_files=self.raw_file_names,
                                      target_dir=self.processed_dir, classes=self.classes,
                                      pre_filter=self.pre_filter, pre_transform=self.pre_transform,
                                      annotations_dir=annotations_dir, read_annotations=read_annotations)

        #########################################################################################################
        # Files #################################################################################################
        #########################################################################################################
        @property
        def raw_file_names(self):
            return glob.glob(os.path.join(self.raw_dir, "*", "*.bin"), recursive=True)

        @property
        def processed_file_names(self):
            return utils.io.read_txt_to_list(os.path.join(self.processed_dir, "files.txt"), default=["unknown"])

        def len(self) -> int:
            return len(self.processed_file_names)

    #########################################################################################################
    # Data Module ###########################################################################################
    #########################################################################################################
    def __init__(self, **kwargs):
        super().__init__(dataset_class=self.NCaltech101DS, **kwargs)

    @property
    def img_shape(self) -> Tuple[int, int]:
        return 240, 180
