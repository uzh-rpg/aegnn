import numpy as np
import torch_geometric
from typing import Callable

from matplotlib import pyplot as plt


def plot(dataset: torch_geometric.data.Dataset, plotting_func: Callable, max_size: int = 100, **kwargs):
    """Plot dataset using some external plotting function, that plots a `torch_geometric.data.Data` object
    in a pyplot axis.

    :param dataset: dataset to plot (if length > 100, sub-sampled).
    :param plotting_func: data plotting function (minimal arguments: data, ax: plt.Axes).
    :param max_size: maximal number of samples to plot, should be multiple of 10 (default = 100).
    :param kwargs: additional (static) arguments for plotting function.
    """
    plot_dataset = dataset
    if len(dataset) > max_size:
        samples = np.random.randint(0, len(dataset), size=max_size).tolist()
        plot_dataset = dataset[samples]
    plot_loader = torch_geometric.data.DataLoader(plot_dataset, batch_size=1, shuffle=False)

    n = len(plot_dataset)
    fig, ax = plt.subplots(5, n // 5, figsize=(10, 2 * n // 5))
    for i, data in enumerate(plot_loader):
        ax_data = ax[i % 5, i // 5]
        ax_data.set_title(getattr(data, "class_id", "unknown"))
        plotting_func(data, ax=ax_data, **kwargs)
    plt.show()
