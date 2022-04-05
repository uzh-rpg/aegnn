import torch


def compute_iou(prediction_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    """Computes for bounding boxes in bbox_detection the IoU with the gt_bbox.

    :param prediction_bbox: [batch_size, gt_bbox_idx, nr_pred_bbox, 4]
    :param gt_bbox: [batch_size, gt_bbox_idx, 4]
    """
    prediction_bbox = torch.clamp(prediction_bbox, min=0).float()
    gt_bbox = torch.clamp(gt_bbox, min=0).float()

    # bbox_detection: bounding_box[0, 0, :]: ['u', 'v', 'w', 'h'].  (u, v) is top left point (u, v) -> (x, y)
    intersection_left_x = torch.max(prediction_bbox[..., 0], gt_bbox[:, None, 0])
    intersection_left_y = torch.max(prediction_bbox[..., 1], gt_bbox[:, None, 1])
    intersection_right_x = torch.min(prediction_bbox[..., 0] + prediction_bbox[..., 2],
                                     gt_bbox[:, None, 0] + gt_bbox[:, None, 2])
    intersection_right_y = torch.min(prediction_bbox[..., 1] + prediction_bbox[..., 3],
                                     gt_bbox[:, None, 1] + gt_bbox[:, None, 3])

    no_intersection_idx = torch.logical_or(torch.gt(intersection_left_y, intersection_right_y),
                                           torch.gt(intersection_left_x, intersection_right_x))
    intersection_area = (intersection_right_x - intersection_left_x) * (intersection_right_y - intersection_left_y)
    intersection_area = torch.max(intersection_area, torch.zeros_like(intersection_area))

    pred_bbox_area = prediction_bbox[..., 2:4].prod(axis=-1)
    gt_bbox_area = gt_bbox[:, None, 2:4].prod(axis=-1)
    union_area = pred_bbox_area + gt_bbox_area - intersection_area

    iou = intersection_area.float() / (union_area.float() + 1e-9)
    iou[no_intersection_idx] = 0
    return iou
