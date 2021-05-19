import numpy as np
import torch
import torch_geometric

from matplotlib import pyplot as plt
from .data import event_histogram


def bbox(batch: torch_geometric.data.Batch, predictions: torch.Tensor = None, max_plots: int = 36):
    """Plot the event graphs stored in the batch (or some of them) as histograms and draw the ground-truth
    detection bounding box(es) above them. If available, the predicted bounding boxes are drawn as well.

    :param batch: batch to draw/sample examples from.
    :param predictions: detected bounding boxes, default = None.
    :param max_plots: maximal number of plots (default = 36), should have an integer square root.
    """
    num_graphs = min(batch.num_graphs, max_plots)
    ax_size = int(np.sqrt(num_graphs))
    assert num_graphs == ax_size * ax_size, "number of plots should be evenly square-able"

    # Get ground-truth bounding boxes and sample random indices to plot.
    bb_gt = getattr(batch, "bbox").view(-1, 1, 5)
    indices = np.random.randint(batch.num_graphs - 1, size=num_graphs)

    # Plot every single graph individually based on functions defined in `data`.
    fig, ax = plt.subplots(ax_size, ax_size, figsize=(ax_size * 5, ax_size * 5))
    for iax, i in enumerate(indices):
        bbox_i = None
        if predictions is not None:
            bbox_i = predictions[i, :].view(1, -1)
        sample = batch.to_data_list()[i]
        sample.bbox = bb_gt[i].view(-1, 5)

        xax = iax // ax_size
        yax = iax % ax_size
        ax[xax, yax] = event_histogram(sample, bbox=bbox_i, ax=ax[xax, yax])
    plt.show()
