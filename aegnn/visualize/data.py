import numpy as np
import torch
import torch_geometric
import tqdm

from typing import List, Tuple, Union
from matplotlib import pyplot as plt

from .utils.bounding_box import draw_bounding_box
from .utils.histogram import compute_histogram
from .utils.title import make_title


def event_histogram(data: torch_geometric.data.Data, img_shape: Tuple[int, int] = None, title: str = None,
                    max_count: int = 1, bbox: torch.Tensor = None, draw_bbox_description: bool = True,
                    ax: plt.Axes = None, **kwargs):
    """Plot event histogram by stacking all events with the same pixel coordinates over all times.

    :param data: sample graph object (pos).
    :param img_shape: image shape in pixel (default = None => inferred from max-xy-coordinate).
    :param title: image title, default = image class.
    :param max_count: maximum count per bin to reject outliers (default = 100, -1 => no outlier rejection).
    :param bbox: bounding boxes to draw additional to data-contained annotation
                 (batch_i, (upper left corner -> u, v), width, height, class_idx, class_conf, prediction_conf)
    :param draw_bbox_description: draw label next to bounding box (default = True).
    :param ax: matplotlib axes to draw in.
    """
    hist = compute_histogram(data.pos[:, :2].cpu().numpy(), img_shape=img_shape, max_count=max_count)
    return image(hist, title, bbox, bbox_gt=getattr(data, "bbox", None), labels_gt=getattr(data, "label", None),
                 draw_bbox_description=draw_bbox_description, ax=ax, **kwargs)


def image(img: Union[torch.Tensor, np.ndarray], title: str = None, bbox: torch.Tensor = None,
          bbox_gt: torch.Tensor = None, labels_gt: List[str] = None, padding: int = 50,
          draw_bbox_description: bool = True, ax: plt.Axes = None
          ) -> plt.Axes:
    """Plot image and ground-truth as well as prediction bounding box (if provided).

    :param img: image to draw.
    :param title: image title, usually a class id (default: unknown).
    :param bbox: prediction bounding boxes (num_bbs, 5), default = None.
    :param bbox_gt: ground-truth bounding box (num_gt_bbs, 5), default = None.
    :param labels_gt: ground-truth labels, default = None.
    :param padding: zero padding around image (default = 50).
    :param draw_bbox_description: draw label next to bounding box (default = True).
    :param ax: matplotlib axes to draw in.
    """
    if not ax:
        _, ax = plt.subplots(1, 1)
    title = make_title(title, default=None)

    img = np.pad(img, pad_width=padding)
    ax.imshow(img.T)
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()

    # If annotations are defined, add the to the plot as a bounding box.
    if bbox_gt is not None:
        assert len(bbox_gt.shape) == 2  # (num_bbs, corner points)
        for i, bounding_box in enumerate(bbox_gt):
            w, h = bounding_box[2:4]
            corner_point = (bounding_box[0], bounding_box[1])
            label = None
            if draw_bbox_description:
                label = labels_gt[i] if labels_gt is not None else ""
                label += f"[{int(bounding_box[-1])}]" if draw_bbox_description else None
            ax = draw_bounding_box(corner_point, w, h, "red", text=label, padding=padding, ax=ax)

    # If bounding boxes are passed to the function, draw them additional to the annotation
    # bounding boxes. Use the prediction confidence as score.
    if bbox is not None:
        assert len(bbox.size()) == 2  # (num_bbs, corner points)
        for i, bbox_i in enumerate(bbox):
            assert bbox_i.numel() == 8
            w, h = bbox_i[3:5]
            corner_point = bbox_i[1:3]
            class_id, class_conf = int(bbox_i[5]), float(bbox_i[6])
            label = f"[{class_id}]({class_conf})" if draw_bbox_description else None
            ax = draw_bounding_box(corner_point, w, h, "green", text=label, padding=padding, ax=ax)

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

    if (edge_index := getattr(data, "edge_index")) is not None:
        for edge in tqdm.tqdm(edge_index.T):
            pos_edge = data.pos[[edge[0], edge[1]], :]
            ax.plot(pos_edge[:, 0], pos_edge[:, 1], "k-", linewidth=0.1)
    ax.scatter(pos_x, pos_y, s=2)
    return ax
