import nestargs
import pytorch_lightning as pl
import torch

import aegnn
import arguments


if __name__ == '__main__':
    parser = nestargs.NestedArgumentParser()
    parser.add_argument("--device", action="store", default=None, type=int)
    parser = arguments.add_dataset_arguments(parser)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device is not None:
        torch.multiprocessing.set_start_method("spawn")
        torch.cuda.set_device(args.device)
        torch.cuda.init()
    pl.seed_everything(args.seed)
    _ = aegnn.datasets.from_args(args)
