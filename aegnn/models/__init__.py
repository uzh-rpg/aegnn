import aegnn.models.base
import aegnn.models.utils

# Object recognition models.
from aegnn.models.nvs import NVS

# Object detection models (YOLO).
from aegnn.models.nvsd import NVSD
from aegnn.models.baseline.cnld import CNLD
from aegnn.models.baseline.rnld import RNLD
from aegnn.models.baseline.yolov4 import YoloV4
from aegnn.models.baseline.asynet import AsyNet

################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl
import typing


def by_name(name: str, **kwargs) -> typing.Union[pl.LightningModule, None]:
    from aegnn.utils.io import select_by_name
    choices = [NVS, NVSD, CNLD, RNLD, YoloV4, AsyNet]
    return select_by_name(choices, name=name, **kwargs)


def get_type(model: typing.Union[pl.LightningModule, None]) -> str:
    if isinstance(model, aegnn.models.base.DetectionModel):
        return "detection"
    elif isinstance(model, aegnn.models.base.RecognitionModel):
        return "recognition"
    return ""

