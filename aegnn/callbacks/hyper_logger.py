import argparse
import pytorch_lightning as pl


class PHyperLogger(pl.callbacks.base.Callback):
    """

    Args:
        args:
        **log_kwargs:
    """

    # More information about the parameters are available in the pytorch-lightning documentation
    # https://pytorch-lightning.readthedocs.io/en/0.7.1/trainer.html
    trainer_keys = ["max_epochs", "min_epochs", "min_steps", "max_steps", "auto_lr_find", "auto_scale_batch_size",
                    "fast_dev_run", "limit_train_batches", "limit_val_batches", "limit_test_batches", "gradient_clip_val",
                    "limit_predict_batches", "overfit_batches", "accumulate_grad_batches", "truncated_bptt_steps"]

    def __init__(self, args: argparse.Namespace, **log_kwargs):
        self.log_kwargs = log_kwargs
        self.log_kwargs.update(vars(args))

    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        # Log trainer hyper-parameters as pre-defined in the callback header.
        trainer_parameters = trainer.__dict__.copy()
        log_parameters = {}
        for key in sorted(self.trainer_keys):
            log_parameters[key] = trainer_parameters.get(key, None)
        model.logger.log_hyperparams(log_parameters)

        # Log number of prediction classes (either from arguments or model).
        num_classes = len(self.log_kwargs.get("classes", []))
        if num_classes == 0:
            num_classes = getattr(model, "num_classes", 0)
        self.log_kwargs["num_classes"] = num_classes

        # Log number of model parameters.
        self.log_kwargs["num_parameters"] = sum(p.numel() for p in model.parameters())
        self.log_kwargs["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Log additional logging kwargs from initialization.
        model.logger.log_hyperparams(self.log_kwargs)
