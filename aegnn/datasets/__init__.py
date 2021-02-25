import aegnn.datasets.utils

from aegnn.datasets.base import EventDataset
from aegnn.datasets.cifar10 import CIFAR10
from aegnn.datasets.ncaltech101 import NCaltech101

################################################################################################
# Access functions #############################################################################
################################################################################################
import argparse
import os
import torch
import typing

import aegnn.filters
import aegnn.transforms


def by_name(name: str, **kwargs) -> typing.Union[EventDataset, None]:
    from aegnn.utils import select_by_name
    return select_by_name([NCaltech101, CIFAR10], name=name, **kwargs)


def from_args(args: argparse.Namespace) -> typing.Union[EventDataset, None]:
    obj_classes = args.tf.classes

    # Parse the pre-transformation class from the name given in the cmd arguments. The name `default`
    # means that the pre-transformation is used that has been stored with the data, i.e. that the
    # processed data is being used like it is.
    if args.tf.pre_transform == "default":
        ptf = get_pre_transform(dataset=args.dataset)
    else:
        ptf = aegnn.transforms.by_name(name=args.tf.pre_transform)

    # Parse the pre-filter class from the name given in the cmd arguments. The name `default`
    # means that the pre-filter is used that has been stored with the data, i.e. that the
    # processed data is being used like it is.
    if args.tf.pre_filter == "default":
        pf, obj_classes = get_pre_filter(dataset=args.dataset)
    else:
        pf = aegnn.filters.by_name(name=args.tf.pre_filter, classes=obj_classes)

    # There are no special cases for the transform, as there isn't preprocessing involved.
    tf = aegnn.transforms.by_name(name=args.tf.transform)

    transform_kwargs = dict(transform=tf, pre_filter=pf, pre_transform=ptf, classes=obj_classes)
    return by_name(args.dataset, **transform_kwargs, **vars(args.data))


def get_pre_filter(dataset: str) -> typing.Tuple[typing.Union[aegnn.filters.Filter, None],
                                                 typing.Union[typing.List[str], None]]:
    path = os.path.join(os.environ["AEGNN_DATA_DIR"], dataset, "training", "processed")
    pf_description = torch.load(os.path.join(path, "pre_filter.pt"))
    pf = aegnn.filters.from_description(pf_description)
    classes = pf.__getattribute__("classes") if hasattr(pf, "classes") else None
    return pf, classes


def get_pre_transform(dataset: str) -> typing.Union[aegnn.transforms.Transform, None]:
    path = os.path.join(os.environ["AEGNN_DATA_DIR"], dataset, "training", "processed")
    ptf_description = torch.load(os.path.join(path, "pre_transform.pt"))
    return aegnn.transforms.from_description(ptf_description)
