import torch

from .pascalvoc import BBType, BoundingBox, BoundingBoxes, MethodAveragePrecision, VOC_Evaluator


def add_pascalvoc_bounding_boxes(bounding_boxes: BoundingBoxes, gt_bbox: torch.Tensor,
                                 detected_bbox: torch.Tensor, image_id: str, image_size: torch.Tensor
                                 ) -> BoundingBoxes:
    """Saves the bounding boxes in the evaluation format
    :param bounding_boxes: bounding boxes to add pascal formatted bbox to.
    :param gt_bbox: gt_bbox: ['u', 'v', 'w', 'h', 'class_id']
    :param detected_bbox: [batch_idx, u, v, w, h, pred_class_id, pred_class_score, object score]
    :param image_id: image identifier.
    """
    image_size = image_size.detach().cpu().numpy()
    for i_batch in range(gt_bbox.shape[0]):
        for i_gt in range(gt_bbox.shape[1]):
            gt_bbox_sample = gt_bbox[i_batch, i_gt, :]
            if gt_bbox[i_batch, i_gt, :].sum() == 0:
                break

            bb_gt = BoundingBox(image_id[i_batch], gt_bbox_sample[-1], gt_bbox_sample[0], gt_bbox_sample[1],
                                gt_bbox_sample[2], gt_bbox_sample[3], image_size, bbType=BBType.GroundTruth,
                                classConfidence=1.0)
            bounding_boxes.addBoundingBox(bb_gt)

    for i_det in range(detected_bbox.shape[0]):
        det_bbox_sample = detected_bbox[i_det, :]
        i_batch = int(det_bbox_sample[0])
        bb_det = BoundingBox(image_id[i_batch], det_bbox_sample[5], det_bbox_sample[1], det_bbox_sample[2],
                             det_bbox_sample[3], det_bbox_sample[4], image_size, bbType=BBType.Detected,
                             classConfidence=det_bbox_sample[6])
        bounding_boxes.addBoundingBox(bb_det)

    return bounding_boxes
