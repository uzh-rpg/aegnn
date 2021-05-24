import numpy as np
import torch

from mapcalc import calculate_map_range
from typing import Dict


def compute_map(gt_bbox: torch.Tensor, detected_bbox: torch.Tensor) -> float:
    """Computes the mean average precision (mAP) score based on the ground-truth and predicted
    bounding boxes, by approximating the area under the precision recall curve by 15 segments.

    :param gt_bbox: gt_bbox: ['u', 'v', 'w', 'h', 'class_id']
    :param detected_bbox: [batch_idx, u, v, w, h, pred_class_id, pred_class_score, object score]
    """
    gt_dict = __parse_gt(gt_bbox)
    detected_dict = __parse_detections(detected_bbox)
    return calculate_map_range(gt_dict, detected_dict, 0.05, 0.95, 0.05)


def __parse_gt(gt_bbox: torch.Tensor) -> Dict[str, np.ndarray]:
    bbox = gt_bbox.detach().cpu().numpy()
    bbox = bbox.reshape(-1, 5)
    return {"boxes": bbox[:, 0:4], "labels": bbox[:, -1]}


def __parse_detections(detected_bbox: torch.Tensor) -> Dict[str, np.ndarray]:
    bbox = detected_bbox.detach().cpu().numpy()
    return {"boxes": bbox[:, 1:5], "labels": bbox[:, 5], "scores": bbox[:, 6]}
