import logging
import os
import pytorch_lightning as pl
import torch


class FullModelCheckpoint(pl.callbacks.ModelCheckpoint):
    FILE_EXTENSION = ".pt"

    def _save_model(self, trainer: pl.Trainer, filepath: str) -> None:
        trainer.dev_debugger.track_checkpointing_history(filepath)
        if trainer.should_rank_save_checkpoint:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(trainer.model, filepath)
        logging.debug(f"Save model checkpoint @ {filepath}")
