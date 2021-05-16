import numpy as np
import torch
import torch_geometric
import tqdm

from typing import Tuple, Union
from matplotlib import pyplot as plt

from .utils.bounding_box import draw_bounding_box
from .utils.histogram import compute_histogram


def event_histogram(data: torch_geometric.data.Data, img_shape: Tuple[int, int] = None,
                    max_count: int = 1, bbox: torch.Tensor = None, ax: plt.Axes = None, return_histogram: bool = False):
    """Plot event histogram by stacking all events with the same pixel coordinates over all times.

    :param data: sample graph object (pos, class_id).
    :param img_shape: image shape in pixel (default = None => inferred from max-xy-coordinate).
    :param max_count: maximum count per bin to reject outliers (default = 100, -1 => no outlier rejection).
    :param bbox: bounding boxes to draw additional to data-contained annotation
                 (batch_i, (upper left corner -> u, v), width, height, class_idx, class_conf, prediction_conf)
    :param ax: matplotlib axes to draw in.
    :param return_histogram: return the 2d histogram next to the axes.
    """
    class_id = getattr(data, "class_id", "unknown")
    hist = compute_histogram(data.pos[:, :2].numpy(), img_shape=img_shape, max_count=max_count)
    ax = image(hist, title=class_id, bbox=bbox, bbox_gt=getattr(data, "bbox", None), ax=ax)
    if return_histogram:
        return ax, hist
    return ax


def image(img: Union[torch.Tensor, np.ndarray], title: str = "unknown", bbox: torch.Tensor = None,
          bbox_gt: torch.Tensor = None, ax: plt.Axes = None) -> plt.Axes:
    """Plot image and ground-truth as well as prediction bounding box (if provided).

    :param img: image to draw.
    :param title: image title, usually a class id (default: unknown).
    :param bbox: prediction bounding boxes (num_bbs, 5), default = None.
    :param bbox_gt: ground-truth bounding box (num_gt_bbs, 5), default = None.
    :param ax: matplotlib axes to draw in.
    """
    if not ax:
        _, ax = plt.subplots(1, 1)

    ax.imshow(img)
    ax.set_title(f"{title}")
    ax.set_axis_off()

    # If annotations are defined, add the to the plot as a bounding box.
    if bbox_gt is not None:
        assert len(bbox_gt.shape) == 2  # (num_bbs, corner points)
        for bounding_box in bbox_gt:
            w, h = bounding_box[2:4]
            corner_point = (bounding_box[1], bounding_box[0])
            ax = draw_bounding_box(corner_point, w, h, color="red", text=title, ax=ax)

    # If bounding boxes are passed to the function, draw them additional to the annotation
    # bounding boxes. Use the prediction confidence as score.
    if bbox is not None:
        assert len(bbox.size()) == 2  # (num_bbs, corner points)
        for bbox_i in bbox:
            assert bbox_i.numel() == 8
            w, h = bbox_i[3:5]
            corner_point = bbox_i[1:3]
            is_correct_class = bbox_i[5] == bbox_gt[-1]  # only one graph, thus direct comparison
            bbox_i_desc = f"{bool(is_correct_class)}({bbox_i[6]:.2f})"
            ax = draw_bounding_box(corner_point, w, h, color="green", text=bbox_i_desc, ax=ax)

    return ax


def graph(data: torch_geometric.data.Data, ax: plt.Axes = None) -> plt.Axes:
    """Plot graph nodes and edges by drawing the nodes as points and connecting them by lines as defined
    in the `edge_index` attribute. Note: This function is quite slow, since the axes plot is called individually
    for each edge.

    :param data: sample graph object (pos, edge_index).
    :param ax: matplotlib axes to draw in.
    """
    if not ax:
        _, ax = plt.subplots(1, 1)

    pos_x = data.pos[:, 0]
    pos_y = data.pos[:, 1]

    edge_index = getattr(data, "edge_index")
    if edge_index is not None:
        for edge in tqdm.tqdm(edge_index.T):
            pos_edge = data.pos[[edge[0], edge[1]], :]
            ax.plot(pos_edge[:, 0], pos_edge[:, 1], "k-", linewidth=0.1)
    ax.plot(pos_x, pos_y, "o")
    return ax
