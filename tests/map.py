import numpy as np
import torch

from aegnn.models.utils.map import compute_map


def compute_random_bboxes(num_bboxes: int, width: int, height: int) -> np.ndarray:
    bboxes = np.zeros((num_bboxes, 4))
    bboxes[:, 0] = np.random.uniform(0, width - 50, size=num_bboxes)  # u
    bboxes[:, 1] = np.random.uniform(0, height - 50, size=num_bboxes)  # v
    bboxes[:, 2] = np.random.uniform(10, width - bboxes[:, 1], size=num_bboxes)  # bbox width
    bboxes[:, 3] = np.random.uniform(10, height - bboxes[:, 2], size=num_bboxes)  # bbox height
    return bboxes


def test_compute_map():
    num_predictions = 10000
    num_gt_bboxes = 500
    class_idx = [i for i in range(5)]
    image_size = 200  # width = height

    predictions = np.zeros((num_predictions, 8))
    predictions[:, 0] = np.random.choice([0, 1, 2], size=num_predictions)  # batch index
    predictions[:, 1:5] = compute_random_bboxes(num_predictions, width=image_size, height=image_size)
    predictions[:, 5] = np.random.choice(class_idx, size=num_predictions)  # class idx
    predictions[:, 6] = np.random.uniform(0, 20, size=num_predictions)  # class score
    predictions[:, 7] = np.random.uniform(0, 20, size=num_predictions)  # object score
    predictions = torch.from_numpy(predictions)

    gt_bboxes = np.zeros((num_gt_bboxes, 5))
    gt_bboxes[:, :4] = compute_random_bboxes(num_gt_bboxes, width=image_size, height=image_size)
    gt_bboxes[:, 4] = np.random.choice(class_idx, size=num_gt_bboxes)  # class idx
    gt_bboxes = torch.from_numpy(gt_bboxes)

    gt_batch = np.random.choice([0, 1, 2], size=num_gt_bboxes)  # batch index
    gt_batch = torch.from_numpy(gt_batch).long()

    # Compute map score for generated prediction and ground-truth bounding boxes. Because of the randomness
    # the resulting score is hard to check, however, at least we can check that the map score is computed
    # in a "feasible" time also for a large number of predictions.
    map_score = compute_map(predictions, gt_bbox=gt_bboxes, gt_batch=gt_batch)
    assert 0 <= map_score <= 1
