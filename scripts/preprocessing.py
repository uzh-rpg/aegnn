import argparse
import pytorch_lightning as pl
import torch

import aegnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", default=None, type=int)
    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        aegnn.utils.loggers.LoggingLogger(None, name="debug")

    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
        if args.num_workers > 1:
            torch.multiprocessing.set_start_method("spawn")
    pl.seed_everything(args.seed)

    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.prepare_data()
