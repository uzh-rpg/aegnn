import functools
import glob
import numpy as np
import os
import torch
import torch_geometric

from typing import List, Tuple, Union

from .base.event_ds import EventDataset
from .base.event_dm import EventDataModule


class NCaltech101(EventDataModule):

    class NCaltech101DS(EventDataset):

        def process(self):
            """Required by `torch_geometric.data.Dataset` in order to start processing."""
            super().process()

        def read_annotations(self, raw_file: str) -> Union[np.ndarray, None]:
            annotations_dir = os.path.join(os.environ["AEGNN_DATA_DIR"], "ncaltech101", "annotations")
            raw_file_rel = os.path.relpath(raw_file, start=self.raw_dir).replace("image", "annotation")

            f = open(os.path.join(os.path.join(annotations_dir, raw_file_rel)))
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
                annotations[1], annotations[0],  # upper-left corner
                annotations[5] - annotations[1],  # width
                annotations[2] - annotations[0],  # height
                class_id
            ])
            bbox[:2] = np.maximum(bbox[:2], 0)
            return bbox.reshape((1, 1, -1))

        def read_label(self, raw_file: str) -> Union[str, List[str], None]:
            return raw_file.split("/")[-2]

        def load(self, raw_file: str) -> torch_geometric.data.Data:
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
            data_obj = torch_geometric.data.Data(x=x, pos=pos)
            return data_obj

        def get_sections(self, data_obj: torch_geometric.data.Data, wdt: float = 0.03
                         ) -> List[torch_geometric.data.Data]:
            t_min = data_obj.pos[:, 2].min()
            t_max = data_obj.pos[:, 2].max()
            data = data_obj.clone()

            # t_start = min(t_min + wdt, t_max - wdt) + random.random() * (t_max - t_min - wdt)
            t_start = t_min + (t_max - wdt - t_min) / 2
            t_end = t_start + wdt
            idx_select = torch.logical_and(t_start <= data.pos[:, 2], data.pos[:, 2] < t_end)
            data.x = data.x[idx_select, :]
            data.pos = data.pos[idx_select, :]
            return [data]

        @functools.lru_cache(maxsize=100)
        def map_label(self, label: str) -> int:
            label_dict = {lbl: i for i, lbl in enumerate(self.classes)}
            return label_dict.get(label, None)

        #########################################################################################################
        # Files #################################################################################################
        #########################################################################################################
        @property
        def raw_files(self):
            return glob.glob(os.path.join(self.raw_dir, "*", "*.bin"), recursive=True)

        @property
        def raw_classes(self) -> List[str]:
            return os.listdir(os.path.join(self.root, "raw"))

        #########################################################################################################
        # Meta ##################################################################################################
        #########################################################################################################
        @property
        def img_shape(self) -> Tuple[int, int]:
            return 240, 240

    #########################################################################################################
    # Data Module ###########################################################################################
    #########################################################################################################
    def __init__(self, **kwargs):
        super().__init__(dataset_class=self.NCaltech101DS, **kwargs)
