import argparse
import datetime
import os
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb

import aegnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model name to train.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform, such as detection or recognition.")
    parser.add_argument("--dim", type=int, help="Dimensionality of input data", default=3)
    parser.add_argument("--seed", default=12345, type=int)

    group = parser.add_argument_group("Trainer")
    group.add_argument("--max-epochs", default=150, type=int)
    group.add_argument("--overfit-batches", default=0.0, type=int)
    group.add_argument("--log-every-n-steps", default=10, type=int)
    group.add_argument("--gradient_clip_val", default=0.0, type=float)
    group.add_argument("--limit_train_batches", default=1.0, type=int)
    group.add_argument("--limit_val_batches", default=1.0, type=int)

    parser.add_argument("--log-gradients", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", default=None, type=int)

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


def main(args):
    log_settings = wandb.Settings(start_method="thread")
    log_dir = os.environ["AEGNN_LOG_DIR"]
    loggers = [aegnn.utils.loggers.LoggingLogger(None if args.debug else log_dir, name="debug")]
    project = f"aegnn-{args.dataset}-{args.task}"
    experiment_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()
    model = aegnn.models.by_task(args.task)(args.model, args.dataset, num_classes=dm.num_classes,
                                            img_shape=dm.dims, dim=args.dim, bias=True, root_weight=True)

    if not args.debug:
        wandb_logger = pl.loggers.WandbLogger(project=project, save_dir=log_dir, settings=log_settings)
        if args.log_gradients:
            wandb_logger.watch(model, log="gradients")  # gradients plot every 100 training batches
        loggers.append(wandb_logger)
    logger = pl.loggers.LoggerCollection(loggers)

    checkpoint_path = os.path.join(log_dir, "checkpoints", args.dataset, args.task, experiment_name)
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        aegnn.utils.callbacks.BBoxLogger(classes=dm.classes),
        aegnn.utils.callbacks.PHyperLogger(args),
        aegnn.utils.callbacks.EpochLogger(),
        aegnn.utils.callbacks.FileLogger([model, model.model, dm]),
        aegnn.utils.callbacks.FullModelCheckpoint(dirpath=checkpoint_path)
    ]

    trainer_kwargs = dict()
    trainer_kwargs["gpus"] = [args.gpu] if args.gpu is not None else None
    trainer_kwargs["profiler"] = "simple" if args.profile else False
    trainer_kwargs["weights_summary"] = "full"
    trainer_kwargs["track_grad_norm"] = 2 if args.log_gradients else -1

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, **trainer_kwargs)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    arguments = parse_args()
    pl.seed_everything(arguments.seed)
    main(arguments)
