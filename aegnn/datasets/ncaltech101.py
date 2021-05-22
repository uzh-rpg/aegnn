import glob
import numpy as np
import os
import torch
import torch_geometric

from typing import Callable, List, Tuple, Union

from .base.event_ds import EventDataset
from .base.event_dm import EventDataModule


class NCaltech101(EventDataModule):

    class NCaltech101DS(EventDataset):

        def __init__(self, root: str, transforms: Callable, pre_transform: Callable, pre_filter: Callable,
                     classes: List[str] = None, mode: str = "training", num_workers: int = 1):
            self.__label_to_class_id = {}
            super().__init__(root, transforms, pre_transform, pre_filter, classes, mode, num_workers)

        def process(self):
            """Required by `torch_geometric.data.Dataset` in order to start processing."""
            super().process()

        def read_annotations(self, raw_file: str) -> np.ndarray:
            annotations_dir = os.path.join(os.environ["AEGNN_DATA_DIR"], "ncaltech101", "annotations")
            raw_file_rel = os.path.relpath(raw_file, start=self.raw_dir).replace("image", "annotation")

            f = open(os.path.join(os.path.join(annotations_dir, raw_file_rel)))
            annotations = np.fromfile(f, dtype=np.int16)
            annotations = np.array(annotations[2:10])
            f.close()

            class_id = self.read_class_id(raw_file)
            return np.array([
                annotations[1], annotations[0],  # upper-left corner
                annotations[5] - annotations[1],  # width
                annotations[2] - annotations[0],  # height
                class_id
            ]).reshape((1, 1, -1))

        def read_class_id(self, raw_file: str) -> Union[int, List[int], None]:
            label = self.read_label(raw_file)
            return self.__label_to_class_id[label]

        def read_label(self, raw_file: str) -> Union[str, List[str], None]:
            label = raw_file.split("/")[-2]
            if label not in self.__label_to_class_id.keys():
                num_entries = len(self.__label_to_class_id)
                self.__label_to_class_id[label] = num_entries
            return label

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

        @property
        def raw_files(self):
            return glob.glob(os.path.join(self.raw_dir, "*", "*.bin"), recursive=True)

    #########################################################################################################
    # Data Module ###########################################################################################
    #########################################################################################################
    def __init__(self, **kwargs):
        super().__init__(dataset_class=self.NCaltech101DS, **kwargs)

    @property
    def img_shape(self) -> Tuple[int, int]:
        return 240, 180
