import torch
import torchvision
from typing import Tuple, Union


def crop_to_frame(bbox: torch.Tensor, image_shape: Union[torch.Tensor, Tuple[int, int]]) -> torch.Tensor:
    """Checks if bounding boxes are inside the image frame of given shape. If not crop it to its border.

    :param bbox: bounding box to check (x, y, width, height).
    :param image_shape: image dimensions (width, height).
    """
    array_width = torch.ones_like(bbox[..., 0], device=bbox.device) * (image_shape[0] - 1)
    array_height = torch.ones_like(bbox[..., 1], device=bbox.device) * (image_shape[1] - 1)
    wh = torch.stack([array_width, array_height], dim=-1)

    xy_delta_min = torch.min(bbox[..., :2], torch.zeros_like(bbox[..., :2], device=bbox.device))
    bbox[..., 0:2] = bbox[..., 0:2] - xy_delta_min
    bbox[..., 2:4] = bbox[..., 2:4] + xy_delta_min

    xy_delta_max = torch.min(wh - bbox[..., :2], torch.zeros_like(bbox[..., :2], device=bbox.device))
    bbox[..., 0:2] = bbox[..., 0:2] + xy_delta_max
    bbox[..., 2:4] = bbox[..., 2:4] - xy_delta_max

    bbox[..., 2] = torch.min(bbox[..., 2], array_width - bbox[..., 0])
    bbox[..., 3] = torch.min(bbox[..., 3], array_height - bbox[..., 1])
    return bbox


def non_max_suppression(detected_bbox: torch.Tensor, iou: float = 0.6) -> torch.Tensor:
    """
    Iterates over the bounding boxes to perform non maximum suppression within each batch.
    :param detected_bbox: [batch_idx, top_left_corner_u,  top_left_corner_v, width, height, predicted_class,
                                 predicted class confidence, object_score])
    :param iou: intersection over union, threshold for which the bbox are considered overlapping
    """
    i_sample = 0
    keep_bbox = []

    while i_sample < detected_bbox.shape[0]:
        same_batch_mask = detected_bbox[:, 0] == detected_bbox[i_sample, 0]
        nms_input = detected_bbox[same_batch_mask][:, [1, 2, 3, 4, 7]].clone()
        nms_input[:, [2, 3]] += nms_input[:, [0, 1]]

        # (u, v) or (x, y) should not matter
        keep_idx = torchvision.ops.nms(nms_input[:, :4], nms_input[:, 4], iou)
        keep_bbox.append(detected_bbox[same_batch_mask][keep_idx])
        i_sample += same_batch_mask.sum()

    if len(keep_bbox) != 0:
        filtered_bbox = torch.cat(keep_bbox, dim=0)
    else:
        filtered_bbox = torch.zeros([0, 8])
    return filtered_bbox
