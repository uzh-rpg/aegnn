import numpy as np
import torch

from mapcalc import calculate_map, calculate_map_range
from typing import Dict, Union


def compute_map(detected_bbox: torch.Tensor, gt_bbox: torch.Tensor, gt_batch: torch.LongTensor,
                iou_threshold: Union[float, np.ndarray] = 0.5) -> float:
    """Computes the mean average precision (mAP) score based on the ground-truth and predicted
    bounding boxes, by approximating the area under the precision recall curve by 15 segments.

    :param detected_bbox: [batch_idx, u, v, w, h, pred_class_idx, pred_class_score, object score]
    :param gt_bbox: [u, v, w, h, class_id]
    :param gt_batch: ground-truth batch index to assign `gt_y` to batch index (samples).
    :param iou_threshold: threshold for defining whether a predicted bounding box is correct.
    """
    assert gt_bbox.size()[0] == gt_batch.numel(), "bbox and batch assignment must have same size"
    batch_indices = torch.unique(gt_batch).long()
    map_scores = np.zeros(batch_indices.numel())

    for i, batch_idx in enumerate(batch_indices):
        gt_dict = __parse_gt(gt_bbox[gt_batch == batch_idx])
        detected_dict = __parse_detections(detected_bbox[detected_bbox[:, 0] == batch_idx])

        if type(iou_threshold) == float:
            map_scores[i] = calculate_map(gt_dict, detected_dict, iou_threshold=iou_threshold)
        else:
            assert len(iou_threshold) >= 2
            iou_step = iou_threshold[1] - iou_threshold[0]
            map_scores[i] = calculate_map_range(gt_dict, detected_dict, iou_threshold[0], iou_threshold[-1], iou_step)

    return float(np.mean(map_scores))


def __parse_gt(gt_bbox: torch.Tensor) -> Dict[str, np.ndarray]:
    bbox = gt_bbox.detach().cpu().numpy()
    bbox = bbox.reshape(-1, 5)
    boxes = np.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2], bbox[:, 1] + bbox[:, 3]], axis=1)
    return {"boxes": boxes, "labels": bbox[:, -1]}


def __parse_detections(detected_bbox: torch.Tensor) -> Dict[str, np.ndarray]:
    bbox = detected_bbox.detach().cpu().numpy()
    boxes = np.stack([bbox[:, 1], bbox[:, 2], bbox[:, 1] + bbox[:, 3], bbox[:, 2] + bbox[:, 4]], axis=1)
    return {"boxes": boxes, "labels": bbox[:, 5], "scores": bbox[:, 6]}
