from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bounding_box(corner_point: Tuple[int, int], width: int, height: int, color: str, text: str, ax: plt.Axes):
    bb_font_dict = dict(multialignment="left", color="black", backgroundcolor=color, size=8)
    rect = patches.Rectangle(corner_point, height, width, linewidth=2, edgecolor=color, facecolor='none')
    ax.text(*corner_point, s=text, fontdict=bb_font_dict)
    ax.add_patch(rect)
    return ax
