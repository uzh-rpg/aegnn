import aegnn.models.layer
import aegnn.models.networks
from aegnn.models.detection import DetectionModel
from aegnn.models.recognition import RecognitionModel

################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl


def by_task(task: str) -> pl.LightningModule.__class__:
    if task == "detection":
        return DetectionModel
    elif task == "recognition":
        return RecognitionModel
    else:
        raise NotImplementedError(f"Task {task} is not implemented!")
