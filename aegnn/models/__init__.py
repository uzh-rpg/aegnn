import aegnn.models.base
import aegnn.models.utils

# Object recognition models.
from aegnn.models.nvs import NVS
from aegnn.models.hist_cnn import HistCNN
from aegnn.models.rnvs import RNVS

# Object detection models (YOLO).
from aegnn.models.nvsd import NVSD

################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl
import typing


def by_name(name: str, **kwargs) -> typing.Union[pl.LightningModule, None]:
    from aegnn.utils.io import select_by_name
    choices = [HistCNN, NVS, NVSD, RNVS]
    return select_by_name(choices, name=name, **kwargs)
