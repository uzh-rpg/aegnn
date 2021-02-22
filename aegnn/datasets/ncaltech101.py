from functools import lru_cache
import os
import numpy as np
import torch
import torch_geometric.data
import tqdm
import typing

import aegnn.utility


class NCaltech101(torch_geometric.data.Dataset):

    def __init__(self, root: str, **dataset_kwargs):
        meta_file = "aegnn/datasets/metadata/ncaltech101.yaml"
        self.__meta_data = aegnn.utility.io.read_yaml_to_dict(meta_file, is_local=True)
        super(NCaltech101, self).__init__(root=root, **dataset_kwargs)

    #########################################################################################################
    # Loading ###############################################################################################
    #########################################################################################################
    def get(self, idx: int) -> torch_geometric.data.Data:
        """Gets the data object at index `idx`.

        :param idx: integer index of (processed) data file to be loaded.
        :return: data object with x = events(x, y, t, p)[N, 4] and y = object-id
        """
        data_file = str(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        obj_class = data_file.split("/")[-2]

        label = self.object_class_to_id(obj_class=obj_class)
        events = torch.load(data_file)
        return torch_geometric.data.Data(x=events, y=label, class_id=obj_class)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    def process(self):
        raw_files = [os.path.join(self.raw_dir, rf) for rf in self.raw_file_names]
        for rf in tqdm.tqdm(raw_files):
            pf = rf.replace(self.raw_dir, self.processed_dir).replace(".bin", ".pt")
            os.makedirs(os.path.dirname(pf), exist_ok=True)
            events = torch.from_numpy(self.__decode_raw_file(raw_file=rf))
            torch.save(events, pf)

    @staticmethod
    def __decode_raw_file(raw_file: str) -> np.ndarray:
        """Decoding method for N-Caltech101 dataset from https://github.com/gorchard/event-Python."""
        f = open(raw_file, "rb")
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_t = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_t[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]
        x = all_x[td_indices]
        y = all_y[td_indices]
        t = all_t[td_indices]
        p = all_p[td_indices]
        return np.array([x, y, t, p]).T.astype(np.float32)

    #########################################################################################################
    # Retrieval #############################################################################################
    #########################################################################################################
    @property
    def raw_file_names(self):
        """The name of the files to find in the `self.raw_dir` folder in order to skip the download."""
        files = []
        for oc, num_files in self.__meta_data.get("object_classes_dict", {}).items():
            files += [os.path.join(self.__meta_data["data_folder"], oc, f"image_{i+1:04d}.bin")
                      for i in range(num_files)]
        return files

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the `self.processed_dir`  folder in order to skip the processing."""
        files = []
        for oc, num_files in self.__meta_data.get("object_classes_dict", {}).items():
            files += [os.path.join(self.__meta_data["data_folder"], oc, f"image_{i+1:04d}.pt")
                      for i in range(num_files)]
        return files

    def download(self):
        """Downloads the dataset to the `self.raw_dir` folder in case it has not been downloaded already."""
        data_url = self.__meta_data.get("data_url", None)
        aegnn.utility.io.download_and_unzip(data_url, folder=self.raw_dir)

    def len(self) -> int:
        """Number of samples in the dataset as the number of files."""
        return len(self.raw_file_names)

    #########################################################################################################
    # Data-Meta-Data ########################################################################################
    #########################################################################################################
    @lru_cache  # caching all results in memory
    def object_class_to_id(self, obj_class: str) -> int:
        obj_class_dict = {oc: i for i, oc in enumerate(self.object_classes)}
        return obj_class_dict[obj_class]

    @property
    def img_shape(self) -> typing.Tuple[int, int]:
        width = self.__meta_data.get("width", -1)
        height = self.__meta_data.get("height", -1)
        return width, height

    @property
    def object_classes(self) -> typing.List[str]:
        return list(self.__meta_data.get("object_classes_dict", {}).keys())
