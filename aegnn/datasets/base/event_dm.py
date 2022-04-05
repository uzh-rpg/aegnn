import abc
import argparse
import logging
import os
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.transforms import Cartesian
from typing import Callable, List, Optional, Tuple

from aegnn.utils.bounding_box import crop_to_frame
from .event_ds import EventDataset


class EventDataModule(pl.LightningDataModule):

    def __init__(self, img_shape: Tuple[int, int], batch_size: int, shuffle: bool, num_workers: int,
                 pin_memory: bool, transform: Optional[Callable[[Data], Data]] = None):
        super(EventDataModule, self).__init__(dims=img_shape)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.transform = transform

    def prepare_data(self) -> None:
        logging.info("Preparing datasets for loading")
        self._prepare_dataset("training")
        self._prepare_dataset("validation")

    def setup(self, stage: Optional[str] = None):
        logging.debug("Load and set up datasets")
        self.train_dataset = self._load_dataset("training")
        self.val_dataset = self._load_dataset("validation")
        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            raise UserWarning("No data found, check AEGNN_DATA_DIR environment variable!")

    #########################################################################################################
    # Data Loaders ##########################################################################################
    #########################################################################################################
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, collate_fn=self.collate_fn,
                                           shuffle=self.shuffle, pin_memory=self.pin_memory)

    def val_dataloader(self, num_workers: int = 2) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, self.batch_size, num_workers=num_workers,
                                           collate_fn=self.collate_fn, shuffle=False)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    @abc.abstractmethod
    def _prepare_dataset(self, mode: str):
        raise NotImplementedError

    @abc.abstractmethod
    def _add_edge_attributes(self, data: Data) -> Data:
        max_value = self.hparams.get("preprocessing", {}).get("r", None)
        edge_attr = Cartesian(norm=True, cat=False, max_value=max_value)
        return edge_attr(data)

    @abc.abstractmethod
    def raw_files(self, mode: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def processed_files(self, mode: str) -> List[str]:
        raise NotImplementedError

    #########################################################################################################
    # Data Loading ##########################################################################################
    #########################################################################################################
    def _load_dataset(self, mode: str):
        processed_files = self.processed_files(mode)
        logging.debug(f"Loaded dataset with {len(processed_files)} processed files")
        return EventDataset(processed_files, load_func=self.load_processed_file)

    def load_processed_file(self, f_path: str) -> Data:
        data = self._load_processed_file(f_path)

        # Post-Processing on loaded data before giving to data loader. Crop and index the bounding boxes
        # and apply the transform if it is defined.
        if hasattr(data, 'bbox'):
            data.bbox = data.bbox.view(-1, 5)
            data.bbox = crop_to_frame(data.bbox, image_shape=self.dims)
        if self.transform is not None:
            data = self.transform(data)

        # Add a default edge attribute, if the data does not have them already.
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            data = self._add_edge_attributes(data)

        # Checking the loaded data for the sake of assuring shape consistency.
        assert data.pos.shape[0] == data.x.shape[0], "x and pos not matching in length"
        assert data.pos.shape[-1] >= 2
        assert data.x.shape[-1] == 1
        assert data.edge_attr.shape[0] == data.edge_index.shape[1], "edges index and attribute not matching"
        assert data.edge_attr.shape[-1] >= 2, "wrong edge attribute dimension"
        if hasattr(data, 'bbox'):
            assert len(data.bbox.shape) == 2 and data.bbox.shape[1] == 5
            assert len(data.y) == data.bbox.shape[0], "annotations not matching"

        return data

    @staticmethod
    def collate_fn(data_list: List[Data]) -> torch_geometric.data.Batch:
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        if hasattr(data_list[0], 'bbox'):
            batch_bbox = sum([[i] * len(data.y) for i, data in enumerate(data_list)], [])
            batch.batch_bbox = torch.tensor(batch_bbox, dtype=torch.long)
        return batch

    @abc.abstractmethod
    def _load_processed_file(self, f_path: str) -> Data:
        """Load pre-processed file to Data object.

        The pre-processed file is loaded into a torch-geometric Data object. With N the number of events,
        L the number of annotations (e.g., bounding boxes in the sample) and P the number of edges, the
        output object should minimally be as shown below.

        :param f_path: input (absolute) file path of preprocessed file.
        :returns Data(x=[N] (torch.float()), pos=[N, 2] (torch.float()), bbox=[L, 5] (torch.long()), file_id,
                      y=[L] (torch.long()), label=[L] (list), edge_index=[2, P] (torch.long())
        """
        raise NotImplementedError

    #########################################################################################################
    # Dataset Properties ####################################################################################
    #########################################################################################################
    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser, **kwargs) -> argparse.ArgumentParser:
        parent_parser.add_argument("--dataset", action="store", type=str, required=True)

        group = parent_parser.add_argument_group("Data")
        group.add_argument("--batch-size", action="store", default=8, type=int)
        group.add_argument("--num-workers", action="store", default=8, type=int)
        group.add_argument("--pin-memory", action="store_true")
        return parent_parser

    @property
    def root(self) -> str:
        return os.path.join(os.environ["AEGNN_DATA_DIR"], self.__class__.__name__.lower())

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def classes(self) -> List[str]:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __repr__(self):
        train_desc = self.train_dataset.__repr__()
        val_desc = self.val_dataset.__repr__()
        return f"{self.__class__.__name__}[Train: {train_desc}\nValidation: {val_desc}]"
