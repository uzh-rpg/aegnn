import numpy as np
import random
import torch
import torch_geometric

from matplotlib import pyplot as plt
from typing import List
from .data import event_histogram


def bbox(batch: torch_geometric.data.Batch, predictions: torch.Tensor = None, titles: List[str] = None,
         max_plots: int = 36, fig_size: int = 2):
    """Plot the event graphs stored in the batch (or some of them) as histograms and draw the ground-truth
    detection bounding box(es) above them. If available, the predicted bounding boxes are drawn as well.

    :param batch: batch to draw/sample examples from.
    :param predictions: detected bounding boxes, default = None.
    :param titles: image titles, default = None, i.e. image class is used.
    :param max_plots: maximal number of plots (default = 36), should have an integer square root.
    :param fig_size: figure size = number of plots per axis * fig_size (default = 2).
    """
    num_graphs = min(batch.num_graphs, max_plots)
    ax_size = int(np.sqrt(num_graphs))
    if titles is not None:
        assert num_graphs <= len(titles)
    assert num_graphs == ax_size * ax_size, "number of plots should be evenly square-able"

    # Get ground-truth bounding boxes and sample random indices to plot.
    bb_gt = getattr(batch, "bbox").view(-1, 1, 5)
    indices = random.sample(list(np.arange(0, batch.num_graphs)), k=num_graphs)

    # Plot every single graph individually based on functions defined in `data`.
    fig, ax = plt.subplots(ax_size, ax_size, figsize=(ax_size * fig_size, ax_size * fig_size))
    for iax, i in enumerate(indices):
        bbox_i = None
        if predictions is not None:
            in_batch = predictions[:, 0] == i
            bbox_i = predictions[in_batch, :]
        title_i = titles[i] if titles is not None else None
        sample = batch.to_data_list()[i]
        sample.bbox = bb_gt[i].view(-1, 5)

        xax = iax // ax_size
        yax = iax % ax_size
        axis = ax[xax, yax] if batch.num_graphs > 1 else ax
        event_histogram(sample, bbox=bbox_i, title=title_i, ax=axis)
    plt.show()
