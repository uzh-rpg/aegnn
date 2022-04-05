import pytorch_lightning as pl


class EpochLogger(pl.callbacks.base.Callback):

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        model.logger.log_metrics({"Epoch": model.current_epoch + 1})
