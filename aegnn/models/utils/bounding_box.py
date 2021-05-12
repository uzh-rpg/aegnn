import torch
import torchvision


def get_grid(input_shape: torch.Tensor, cell_map_shape: torch.Tensor) -> torch.Tensor:
    """Constructs a 2D grid with the cell center coordinates."""
    cell_shape = input_shape / cell_map_shape
    num_cells = (cell_map_shape * cell_shape).int()
    cell_top_left = torch.meshgrid([torch.arange(0, end=num_cells[0], step=cell_shape[0], device=cell_shape.device),
                                    torch.arange(0, end=num_cells[1], step=cell_shape[1], device=cell_shape.device)])
    return torch.stack(cell_top_left, dim=-1)


def crop_to_frame(bbox: torch.Tensor, image_shape: torch.Tensor) -> torch.Tensor:
    """Checks if bounding boxes are inside frame. If not crop to border"""
    array_width = torch.ones_like(bbox[:, 1]) * image_shape[1] - 1
    array_height = torch.ones_like(bbox[:, 2]) * image_shape[0] - 1

    bbox[:, 1:3] = torch.max(bbox[:, 1:3], torch.zeros_like(bbox[:, 1:3]))
    bbox[:, 1] = torch.min(bbox[:, 1], array_width)
    bbox[:, 2] = torch.min(bbox[:, 2], array_height)

    bbox[:, 3] = torch.min(bbox[:, 3], array_width - bbox[:, 1])
    bbox[:, 4] = torch.min(bbox[:, 4], array_height - bbox[:, 2])

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
