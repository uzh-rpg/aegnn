import itertools

import numpy as np
from matplotlib import pyplot as plt


def confusion_matrix(cm, target_names, title="Confusion matrix", colormap=None, normalize=True, close: bool = False):
    """Plot a sklearn confusion matrix (cm).

    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: given classification classes such as [0, 1, 2] the class names.
    :param title: the text to display at the top of the matrix
    :param colormap: the gradient of the values displayed from matplotlib.pyplot.cm,
                     see http://matplotlib.org/examples/color/colormaps_reference.html
                     e.g. plt.get_cmap('jet') or plt.cm.Blues
    :param normalize: If False, plot the raw numbers. Otherwise, plot the proportions
    :param close: close plot before returning (for storing, not displaying).
    Citation
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = cm.numpy()
    if colormap is None:
        colormap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=colormap)
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm += 1e-6  # avoiding divisions by zero
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = "{:0.4f}".format(cm[i, j]) if normalize else "{:,}".format(cm[i, j])
        color = colormap(256) if cm[i, j] < thresh else colormap(0)
        plt.text(j, i, value, horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if close:
        plt.close(fig)
    return fig
