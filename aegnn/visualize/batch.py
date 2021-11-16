import numpy as np
import random
import torch
import torch_geometric

from matplotlib import pyplot as plt
from typing import List
from .data import event_histogram


def bbox(batch: torch_geometric.data.Batch, predictions: torch.Tensor = None, titles: List[str] = None,
         max_plots: int = 36, draw_bbox_description: bool = True, fig_size: int = 2, max_plots_row: int = 4):
    """Plot the event graphs stored in the batch (or some of them) as histograms and draw the ground-truth
    detection bounding box(es) above them. If available, the predicted bounding boxes are drawn as well.

    :param batch: batch to draw/sample examples from.
    :param predictions: detected bounding boxes, default = None.
    :param titles: image titles, default = None, i.e. image class is used.
    :param max_plots: maximal number of plots (default = 36), should have an integer square root.
    :param draw_bbox_description: draw label next to bounding box (default = True).
    :param fig_size: figure size = number of plots per axis * fig_size (default = 2).
    :param max_plots_row: maximal number of plots per row (default = 4).
    """
    num_graphs = min(batch.num_graphs, max_plots)
    if titles is not None:
        assert num_graphs <= len(titles)

    # Get ground-truth bounding boxes and sample random indices to plot.
    bb_gt = getattr(batch, "bbox")
    bb_gt_index = getattr(batch, "batch_bbox")
    indices = random.sample(list(np.arange(0, batch.num_graphs)), k=num_graphs)

    # Adapt the plot distribution to the number of graphs.
    if num_graphs < max_plots_row:
        ax_size_x = 1
        ax_size_y = num_graphs
    else:
        ax_size_y = max_plots_row
        ax_size_x = num_graphs // max_plots_row

    # Plot every single graph individually based on functions defined in `data`.
    fig, ax = plt.subplots(ax_size_x, ax_size_y, figsize=(ax_size_y * fig_size, ax_size_x * fig_size))
    for iax, i in enumerate(indices):
        bbox_i = None
        if predictions is not None:
            in_batch = predictions[:, 0] == i
            bbox_i = predictions[in_batch, :]
        title_i = titles[i] if titles is not None else None
        sample = batch.to_data_list()[i]
        sample.bbox = bb_gt[bb_gt_index == i].view(-1, 5)

        axis = __get_axis(ax, iax, num_graphs=num_graphs, max_plots_row=max_plots_row)
        event_histogram(sample, bbox=bbox_i, title=title_i, draw_bbox_description=draw_bbox_description, ax=axis)
    plt.show()


def __get_axis(ax: plt.Axes, iax: int, num_graphs: int, max_plots_row: int):
    xax = iax // max_plots_row
    yax = iax % max_plots_row
    if num_graphs == 1:
        axis = ax
    elif num_graphs <= max_plots_row:
        axis = ax[yax]
    else:
        axis = ax[xax, yax]
    return axis
