import aegnn.models.base

from aegnn.models.nvs import NVS
from aegnn.models.hist_cnn import HistCNN
from aegnn.models.rnvs import RNVS


################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl
import typing


def by_name(name: str, **kwargs) -> typing.Union[pl.LightningModule, None]:
    from aegnn.utils import select_by_name
    choices = [HistCNN, NVS, RNVS]
    return select_by_name(choices, name=name, **kwargs)
