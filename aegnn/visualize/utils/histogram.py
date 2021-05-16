import logging
import numpy as np

from typing import Tuple


def compute_histogram(positions: np.ndarray, img_shape: Tuple[int, int] = None, max_count: int = -1) -> np.ndarray:
    """Create histogram from two-dimensional (event) data points positions x by bucketing them into discrete pixels.

    :param positions: data point position tensor (N, >=2).
    :param img_shape: output image shape. If `None` the coordinate-wise maxima will be used.
    :param max_count: maximum count per bin to reject outliers (default = 100, -1 => no outlier rejection).
    :returns: images stacked over batches (N_batches, H, W).
    """
    x_min, y_min = positions[:, :2].min(axis=0).astype(int)
    if img_shape is None:
        img_shape = positions[:, :2].max(axis=0).astype(int)
        logging.debug(f"Inferred image shape to be {img_shape}")
    x_max, y_max = img_shape

    # Plot the spatial component of the events stacked in a 2d histogram. For outlier rejection reset
    # bins with very high event count to a lower threshold.
    x_bins = np.linspace(x_min, x_max, num=x_max + 1)
    y_bins = np.linspace(y_min, y_max, num=y_max + 1)
    histogram, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=(x_bins, y_bins))

    # Limit the number of counts per pixel, when also lower count pixel should be highlighted.
    if max_count > 0:
        histogram[histogram > max_count] = max_count
    return histogram
