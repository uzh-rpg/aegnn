import os
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader
from typing import List, Optional, Tuple, Union

from aegnn.utils.filters import Filter
from aegnn.utils.transforms import Transform
from .event_ds import EventDataset


class EventDataModule(pl.LightningDataModule):

    def __init__(self, dataset_class: EventDataset.__class__, transforms=None, pre_transform=None, pre_filter=None,
                 classes: List[str] = None, batch_size: int = 64, shuffle: bool = True,
                 num_workers: int = 8, pin_memory: bool = False):
        super().__init__()

        self.__kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        dataset_kwargs = dict(pre_transform=pre_transform, pre_filter=pre_filter, root=self.root,
                              classes=classes, num_workers=num_workers)

        transforms = transforms if (transforms is not None and all(transforms)) else []
        self.train_dataset = dataset_class(mode="training", transforms=transforms, **dataset_kwargs)
        self.val_dataset = dataset_class(mode="validation", transforms=transforms, **dataset_kwargs)

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        pass

    #########################################################################################################
    # Data Loaders ##########################################################################################
    #########################################################################################################
    def train_dataloader(self) -> DataLoader:
        subset = self.train_dataset.get_subset(label=self.classes)
        return DataLoader(subset, collate_fn=self.train_dataset.collate, **self.__kwargs)

    def val_dataloader(self) -> DataLoader:
        subset = self.val_dataset.get_subset(label=self.classes)
        batch_size = min(self.__kwargs.get("batch_size", 1), len(subset))
        return DataLoader(subset, collate_fn=self.val_dataset.collate, batch_size=batch_size,
                          num_workers=2, shuffle=False)

    #########################################################################################################
    # Transform Properties ##################################################################################
    #########################################################################################################
    @property
    def transforms(self) -> List[Transform]:
        return self.train_dataset.transform.transforms

    @property
    def pre_transform(self) -> Union[Transform, None]:
        return self.train_dataset.pre_transform

    @property
    def pre_filter(self) -> Union[Filter, None]:
        return self.train_dataset.pre_filter

    #########################################################################################################
    # Dataset Properties ####################################################################################
    #########################################################################################################
    @property
    def root(self) -> str:
        return os.path.join(os.environ["AEGNN_DATA_DIR"], self.__class__.__name__.lower())

    @property
    def classes(self) -> List[str]:
        return self.train_dataset.classes

    @property
    def num_classes(self) -> int:
        return len(self.train_dataset.classes)

    @property
    def img_shape(self) -> Tuple[int, int]:
        return self.train_dataset.img_shape

    def __repr__(self):
        train_desc = self.train_dataset.__repr__()
        val_desc = self.val_dataset.__repr__()
        return f"{self.__class__.__name__}[Train: {train_desc}\nValidation: {val_desc}]"
