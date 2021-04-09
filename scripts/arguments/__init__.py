import argparse
from nestargs.parser import NestedArgumentParser


def add_dataset_arguments(parser: NestedArgumentParser) -> NestedArgumentParser:
    parser.add_argument("--dataset", action="store", default="ncaltech101", type=str)
    parser.add_argument("--seed", action="store", default=12345, type=int)
    parser.add_argument("--debug", action="store_true")

    group = parser.add_argument_group("data")
    group.add_argument("--data.batch-size", action="store", default=64, type=int)
    group.add_argument("--data.num-workers", action="store", default=8, type=int)
    group.add_argument("--data.shuffle", action="store", default=True, type=lambda x: x.lower() == "true")
    group.add_argument("--data.pin-memory", action="store_true")

    group = parser.add_argument_group("tf")
    group.add_argument("--tf.transforms", action="store", default=[], type=lambda x: x.split(":"))
    group.add_argument("--tf.pre-transform", action="store", default="default", type=str)
    group.add_argument("--tf.pre-filter", action="store", default="default", type=str)
    group.add_argument("--tf.classes", action="store", type=lambda x: x.split(":"), default=[])

    return parser


def add_trainer_arguments(parser: NestedArgumentParser) -> NestedArgumentParser:
    group = parser.add_argument_group("train")
    group.add_argument("--train.max-epochs", action="store", default=150, type=int)
    group.add_argument("--train.overfit-batches", action="store", default=0.0, type=int)
    group.add_argument("--train.log-gradients", action="store_true")
    group.add_argument("--train.log-steps", action="store", default=10, type=int)
    group.add_argument("--train.gradient-clipping", action="store", default=0.0, type=float)

    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gpu", action="store", default=None, type=int)
    parser.add_argument("--logging", action="store", default=True, type=lambda x: x.lower() == "true")

    return parser


def check_arguments(args: argparse.Namespace):
    if args.train.overfit_batches > 0 and args.data.shuffle:
        raise ValueError("Shuffling and over-fitting not compatible, add --data.shuffle=False!")
