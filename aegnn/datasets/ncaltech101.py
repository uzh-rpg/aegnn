import glob
import numpy as np
import os
import torch
import torch_geometric

from typing import Callable, List, Tuple, Optional

import aegnn.datasets.utils as utils
from aegnn.utils import TaskManager
from .base.event_ds import EventDataset


class NCaltech101(EventDataset):

    class NCaltech101DS(torch_geometric.data.Dataset):

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
        def process(self):
            annotations_dir = os.path.join(os.environ["AEGNN_DATA_DIR"], "ncaltech101", "annotations")

            print("Building class dictionary")
            object_class_ids = utils.data.build_class_dict(self.raw_files, classes=self.classes,
                                                           read_class_id=self.read_class_id)

            # Processing the raw files in parallel processes. Importantly, the functions must therefore be able to be
            # pickled, i.e. not using dynamic types or not-shared class variables.
            print(f"Processing raw files with {self.num_workers} workers")
            task_manager = TaskManager(self.num_workers, queue_size=10, callback=None, total=len(self.raw_files))
            for rf in self.raw_file_names:
                task_manager.queue(utils.data.processing, rf=rf, raw_dir=self.raw_dir, target_dir=self.target_dir,
                                   object_class_ids=object_class_ids, load_func=self.load, wdt=0.03,
                                   pre_filter=self.pre_filter, pre_transform=self.pre_transform,
                                   annotations_dir=annotations_dir, read_annotations=self.read_annotations)

        @staticmethod
        def read_annotations(annotation_file: str) -> np.ndarray:
            file_name = os.path.basename(annotation_file).replace("image", "annotation")
            f = open(os.path.join(os.path.dirname(annotation_file), file_name))
            annotations = np.fromfile(f, dtype=np.int16)
            f.close()
            return np.array(annotations[2:10]).reshape(1, -1)

        @staticmethod
        def read_class_id(raw_file: str) -> str:
            return raw_file.split("/")[-2]

        @staticmethod
        def load(raw_file: str) -> torch_geometric.data.Data:
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

            obj_class = raw_file.split("/")[-2]
            file_id = raw_file.split("/")[-1]
            x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
            data_obj = torch_geometric.data.Data(x=x, pos=pos, class_id=obj_class, file_id=file_id)
            return data_obj

        #########################################################################################################
        # Files #################################################################################################
        #########################################################################################################
        @property
        def raw_file_names(self):
            return glob.glob(os.path.join(self.raw_dir, "*", "*.bin"), recursive=True)

        @property
        def processed_file_names(self):
            files = glob.glob(os.path.join(self.raw_dir, "*", "*.pt"), recursive=True)
            if len(files) == 0:
                return ["unknown"]
            return files

        def download(self):
            pass

        def len(self) -> int:
            return len(self.processed_file_names)

    #########################################################################################################
    # Data Module ###########################################################################################
    #########################################################################################################
    def __init__(self, **kwargs):
        super().__init__(dataset_class=self.NCaltech101DS, **kwargs)

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        pass

    @property
    def img_shape(self) -> Tuple[int, int]:
        return 240, 180
