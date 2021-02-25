import nestargs
import os
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb

import aegnn
import arguments


if __name__ == '__main__':
    parser = nestargs.NestedArgumentParser()
    parser.add_argument("model", action="store", default=None, type=str)
    parser = arguments.add_dataset_arguments(parser)
    parser = arguments.add_trainer_arguments(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    arguments.check_arguments(args)

    project_name = f"aegnn-{args.dataset}"
    log_settings = wandb.Settings(start_method="thread")
    log_dir = os.environ["AEGNN_LOG_DIR"]

    dm = aegnn.datasets.from_args(args)
    model = aegnn.models.by_name(args.model, num_classes=dm.num_classes, img_shape=dm.img_shape)
    logger = pl.loggers.WandbLogger(project=project_name, save_dir=log_dir, settings=log_settings, sync_step=True)
    if args.train.log_gradients:
        logger.watch(model, log="all", log_freq=1)

    model_dir = os.path.join(log_dir, "models", logger.experiment.name)
    log_cb = aegnn.callbacks.PHyperLogger(args, objects=[model, dm.pre_filter, dm.pre_transform, dm.transform])
    confusion_matrix_cb = aegnn.callbacks.ConfusionMatrix(classes=dm.classes)
    early_stopping_cb = pl.callbacks.EarlyStopping(monitor="Val/Loss", mode="min")
    learning_rate_cb = pl.callbacks.LearningRateMonitor()
    checkpoint_cb = pl.callbacks.ModelCheckpoint(dirpath=model_dir, save_top_k=1, monitor="Val/Loss", mode="min")
    callbacks = [log_cb, confusion_matrix_cb, checkpoint_cb, learning_rate_cb]

    trainer_kwargs = dict()
    trainer_kwargs["gpus"] = [args.gpu] if args.gpu else None
    trainer_kwargs["profiler"] = args.profile
    trainer_kwargs["precision"] = 16 if args.gpu else 32
    trainer_kwargs["max_epochs"] = args.train.max_epochs
    trainer_kwargs["overfit_batches"] = args.train.overfit_batches
    trainer_kwargs["weights_summary"] = "full"
    trainer_kwargs["gradient_clip_val"] = args.train.gradient_clipping
    trainer_kwargs["track_grad_norm"] = 2 if args.train.log_gradients else -1
    trainer_kwargs["log_every_n_steps"] = args.train.log_steps

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)
    trainer.fit(model, datamodule=dm)
