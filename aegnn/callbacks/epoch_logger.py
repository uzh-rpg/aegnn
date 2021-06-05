import pytorch_lightning as pl
import pytorch_lightning.loggers


class EpochLogger(pl.callbacks.base.Callback):

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if isinstance(model.logger, pytorch_lightning.loggers.WandbLogger):
            model.logger.experiment.log({"Epoch": trainer.current_epoch}, commit=False)
