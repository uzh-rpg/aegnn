from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bounding_box(corner_point: Tuple[int, int], width: int, height: int, color: str,
                      padding: int, ax: plt.Axes, text: str = None):
    bb_font_dict = dict(multialignment="left", color="black", backgroundcolor=color, size=8)
    corner_point = (corner_point[0] + padding, corner_point[1] + padding)
    rect = patches.Rectangle(corner_point, width, height, linewidth=2, edgecolor=color, facecolor='none')
    if text is not None:
        ax.text(*corner_point, s=text, fontdict=bb_font_dict)
    ax.add_patch(rect)
    return ax
