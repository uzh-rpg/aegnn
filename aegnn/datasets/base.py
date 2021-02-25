import os
import pytorch_lightning as pl

from torch_geometric.data import Dataset, DataLoader
from typing import List, Tuple, Union

from aegnn.filters import Filter
from aegnn.transforms import Transform


class EventDataset(pl.LightningDataModule):

    def __init__(self, dataset_class: Dataset.__class__, transform=None, pre_transform=None, pre_filter=None,
                 classes: List[str] = None, batch_size: int = 64, shuffle: bool = True,
                 num_workers: int = 8, pin_memory: bool = False):
        tf_train = [transform] if transform is not None else None
        super().__init__(train_transforms=tf_train)

        self.__kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        dataset_kwargs = dict(pre_transform=pre_transform, pre_filter=pre_filter, root=self.root, classes=classes)

        self.train_dataset = dataset_class(mode="training", transform=None, **dataset_kwargs)
        self.val_dataset = dataset_class(mode="validation", transform=None, **dataset_kwargs)
        self.test_dataset = dataset_class(mode="testing", transform=None, **dataset_kwargs)

    #########################################################################################################
    # Data Transformers #####################################################################################
    #########################################################################################################
    @staticmethod
    def dataset_on_each(dataset: Dataset) -> Dataset:
        indices, class_added = [], []

        for idx, sample in enumerate(dataset):
            y = sample.y
            if y not in class_added:
                class_added.append(y)
                indices.append(idx)
            if len(class_added) == len(dataset.classes):
                break

        return dataset.index_select(indices)

    #########################################################################################################
    # Data Loaders ##########################################################################################
    #########################################################################################################
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.__kwargs)

    def val_dataloader(self) -> DataLoader:
        batch_size = self.__kwargs.get("batch_size", 1)
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        batch_size = self.__kwargs.get("batch_size", 1)
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    #########################################################################################################
    # Transform Properties ##################################################################################
    #########################################################################################################
    @property
    def transform(self) -> Union[Transform, None]:
        return self.train_transforms

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
        raise NotImplementedError

    def __repr__(self):
        train_desc = self.train_dataset.__repr__()
        val_desc = self.val_dataset.__repr__()
        test_desc = self.test_dataset.__repr__()
        return f"{self.__class__.__name__}[Train: {train_desc}\nValidation: {val_desc}\nTesting: {test_desc}]"
