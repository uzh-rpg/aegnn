from aegnn.datasets.base.event_dm import EventDataModule
from aegnn.datasets.ncaltech101 import NCaltech101
from aegnn.datasets.megapixel import Megapixel

################################################################################################
# Access functions #############################################################################
################################################################################################
import argparse
import os
import torch
from typing import List, Tuple, Union

import aegnn.utils.filters
import aegnn.utils.transforms


def by_name(name: str, **kwargs) -> Union[EventDataModule, None]:
    from aegnn.utils.io import select_by_name
    return select_by_name([NCaltech101, Megapixel], name=name, **kwargs)


def from_args(args: argparse.Namespace) -> Union[EventDataModule, None]:
    classes_selected = args.tf.classes

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
        pf, classes_selected_pf = get_pre_filter(dataset=args.dataset)
        if len(classes_selected) == 0:
            classes_selected = classes_selected_pf
    else:
        pf = aegnn.utils.filters.by_name(name=args.tf.pre_filter, labels=classes_selected)

    # There are no special cases for the transform, as there isn't preprocessing involved.
    tfs = [aegnn.utils.transforms.by_name(name=tf) for tf in args.tf.transforms]
    transform_kwargs = dict(transforms=tfs, pre_filter=pf, pre_transform=ptf, classes=classes_selected)
    return by_name(args.dataset, **transform_kwargs, **vars(args.data))


def get_pre_filter(dataset: str) -> Tuple[Union[aegnn.utils.filters.Filter, None],
                                          Union[List[str], None]]:
    path = os.path.join(os.environ["AEGNN_DATA_DIR"], dataset, "training", "processed")
    pf_description = torch.load(os.path.join(path, "pre_filter.pt"))
    pre_filter = aegnn.utils.filters.from_description(pf_description)
    return pre_filter, getattr(pre_filter, "labels", None)


def get_pre_transform(dataset: str) -> Union[aegnn.utils.transforms.Transform, None]:
    path = os.path.join(os.environ["AEGNN_DATA_DIR"], dataset, "training", "processed")
    ptf_description = torch.load(os.path.join(path, "pre_transform.pt"))
    return aegnn.utils.transforms.from_description(ptf_description)
