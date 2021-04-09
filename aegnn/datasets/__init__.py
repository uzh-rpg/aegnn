import aegnn.datasets.utils

from aegnn.datasets.base.event_ds import EventDataset
from aegnn.datasets.ncaltech101 import NCaltech101

################################################################################################
# Access functions #############################################################################
################################################################################################
import argparse
import os
import torch
import typing

import aegnn.utils.filters
import aegnn.utils.transforms


def by_name(name: str, **kwargs) -> typing.Union[EventDataset, None]:
    from aegnn.utils.io import select_by_name
    return select_by_name([NCaltech101], name=name, **kwargs)


def from_args(args: argparse.Namespace) -> typing.Union[EventDataset, None]:
    obj_classes = args.tf.classes

    # Parse the pre-transformation class from the name given in the cmd arguments. The name `default`
    # means that the pre-transformation is used that has been stored with the data, i.e. that the
    # processed data is being used like it is.
    if args.tf.pre_transform == "default":
        ptf = get_pre_transform(dataset=args.dataset)
    else:
        ptf = aegnn.utils.transforms.by_name(name=args.tf.pre_transform)

    # Parse the pre-filter class from the name given in the cmd arguments. The name `default`
    # means that the pre-filter is used that has been stored with the data, i.e. that the
    # processed data is being used like it is.
    if args.tf.pre_filter == "default":
        pf, obj_classes = get_pre_filter(dataset=args.dataset)
    else:
        pf = aegnn.utils.filters.by_name(name=args.tf.pre_filter, classes=obj_classes)

    # There are no special cases for the transform, as there isn't preprocessing involved.
    tfs = [aegnn.utils.transforms.by_name(name=tf) for tf in args.tf.transforms]

    transform_kwargs = dict(transforms=tfs, pre_filter=pf, pre_transform=ptf, classes=obj_classes)
    return by_name(args.dataset, **transform_kwargs, **vars(args.data))


def get_pre_filter(dataset: str) -> typing.Tuple[typing.Union[aegnn.utils.filters.Filter, None],
                                                 typing.Union[typing.List[str], None]]:
    path = os.path.join(os.environ["AEGNN_DATA_DIR"], dataset, "training", "processed")
    pf_description = torch.load(os.path.join(path, "pre_filter.pt"))
    pf = aegnn.utils.filters.from_description(pf_description)
    classes = pf.__getattribute__("classes") if hasattr(pf, "classes") else None
    return pf, classes


def get_pre_transform(dataset: str) -> typing.Union[aegnn.utils.transforms.Transform, None]:
    path = os.path.join(os.environ["AEGNN_DATA_DIR"], dataset, "training", "processed")
    ptf_description = torch.load(os.path.join(path, "pre_transform.pt"))
    return aegnn.utils.transforms.from_description(ptf_description)
