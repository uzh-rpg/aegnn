import torch


def compute_detection_accuracy(detected_bbox: torch.Tensor, gt_y: torch.Tensor, gt_batch: torch.LongTensor,
                               threshold: float = 0.5) -> float:
    """Compute the recognition accuracy for a detection task over a batch.

    The recognition accuracy is determined by iterating over the ground-truth labels in each sample (=batch unit) and
    checking whether there are any predictions with a probability of having the same class larger than the
    recognition threshold.

    :param detected_bbox: [batch_idx, u, v, w, h, pred_class_idx, pred_class_score, object score]
    :param gt_y: ground-truth class index over the batch.
    :param gt_batch: ground-truth batch index to assign `gt_y` to batch index (samples).
    :param threshold: prediction class probability threshold (default = 0.5).
    """
    assert gt_y.numel() == gt_batch.numel(), "labels and batch assignment must have same size"
    true_positives = 0
    count = 0

    batch_indices = torch.unique(gt_batch).long()
    for batch_idx in batch_indices:
        gt_y_masked = gt_y[gt_batch == batch_idx].long()
        det_masked = detected_bbox[detected_bbox[:, 0] == batch_idx, :].long()

        for gt_yy in gt_y_masked:
            det_masked_yy_probability = det_masked[det_masked[:, 5] == gt_yy, 6]
            if torch.any(det_masked_yy_probability > threshold):
                true_positives += 1
            count += 1

    if count == 0:
        return 1.0
    return true_positives / count
