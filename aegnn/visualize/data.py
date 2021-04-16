import logging
import numpy as np
import torch_geometric
import tqdm

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def event_histogram(data: torch_geometric.data.Data, img_width: float = None, img_height: float = None,
                    max_count: int = 1, ax: plt.Axes = None, return_histogram: bool = False):
    """Plot event histogram by stacking all events with the same pixel coordinates over all times.

    :param data: sample graph object (pos, class_id).
    :param img_width: image width in pixel (default = None => inferred from max-x-coordinate).
    :param img_height: image height in pixel (default = None => inferred from max-y-coordinate).
    :param max_count: maximum count per bin to reject outliers (default = 100, -1 => no outlier rejection).
    :param ax: matplotlib axes to draw in.
    :param return_histogram: return the 2d histogram next to the axes.
    """
    if not ax:
        _, ax = plt.subplots(1, 1)
    xy = data.pos[:, :2].numpy()
    class_id = getattr(data, "class_id", "unknown")

    # Although the image dimensions are not given, they are still required for plotting the histogram.
    # Therefore, if they are not given, it is assumed that the coordinates are in between [0, img_width]
    # while the `img_width` is approximated by the maximum value in the x-dimension (similar for y-direction).
    x_min, y_min = xy[:, :2].min(axis=0)
    if not img_width:
        img_width = int(xy[:, 0].max())
        logging.debug(f"Inferred image width to be {img_width}")
    if not img_height:
        img_height = int(xy[:, 1].max())
        logging.debug(f"Inferred image height to be {img_height}")

    # Plot the spatial component of the events stacked in a 2d histogram. For outlier rejection reset
    # bins with very high event count to a lower threshold.
    x_bins = np.linspace(x_min, img_width, num=img_width + 1)
    y_bins = np.linspace(y_min, img_height, num=img_height + 1)
    histogram, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_bins, y_bins))
    if max_count > 0:
        histogram[histogram > max_count] = max_count

    ax.imshow(histogram)
    ax.set_title(f"{class_id}")
    ax.set_axis_off()

    # If annotations are defined, add the to the plot as a bounding box.
    bounding_boxes = getattr(data, "bb", None)
    if bounding_boxes is not None:
        assert len(bounding_boxes.shape) == 2  # (num_bbs, corner points)
        bb_font_dict = dict(multialignment="left", color="black", backgroundcolor="red")
        for bounding_box in bounding_boxes:
            corner_point = (bounding_box[0], bounding_box[1])
            h = bounding_box[2] - bounding_box[0]
            w = bounding_box[5] - bounding_box[1]
            rect = patches.Rectangle(corner_point, w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.text(*corner_point, s=class_id, fontdict=bb_font_dict)
            ax.add_patch(rect)

    if return_histogram:
        return ax, histogram
    return ax


def graph(data: torch_geometric.data.Data, ax: plt.Axes = None):
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
    ax.plot(pos_x, pos_y, "o")

    edge_index = getattr(data, "edge_index")
    if edge_index is not None:
        for edge in tqdm.tqdm(edge_index.T):
            pos_edge = data.pos[[edge[0], edge[1]], :]
            ax.plot(pos_edge[:, 0], pos_edge[:, 1], "-")
