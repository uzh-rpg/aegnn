"""Partly copied from rpg-asynet paper: https://github.com/uzh-rpg/rpg_asynet"""
import collections
import logging
import numpy as np
import torch
import torch_geometric
import pytorch_lightning as pl

from typing import Dict, Tuple

from aegnn.utils.bounding_box import crop_to_frame, non_max_suppression
from .utils.accuracy import compute_detection_accuracy
from .utils.iou import compute_iou
from .utils.map import compute_map
from .utils.yolo import yolo_grid
from .networks import by_name as model_by_name


class DetectionModel(pl.LightningModule):
    def __init__(self, network: str, dataset: str, num_classes: int, img_shape: Tuple[int, int], dim: int = 3,
                 num_bounding_boxes: int = 1, **model_kwargs):
        super(DetectionModel, self).__init__()

        # Define the YOLO detection grid as the model's outputs.
        self.num_classes = num_classes
        self.num_bounding_boxes = num_bounding_boxes
        self.cell_map_shape = (8, 6)
        self.input_shape = torch.tensor(img_shape, device=self.device)
        self.dim = dim

        self.num_outputs_per_cell = num_classes + num_bounding_boxes * 5  # (x, y, width, height, confidence)
        num_outputs = self.num_outputs_per_cell * self.cell_map_shape[0] * self.cell_map_shape[1]  # detection grid

        # Define network architecture by name.
        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        self.model = model_by_name(network)(dataset, model_input_shape, num_outputs=num_outputs, **model_kwargs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.pos = data.pos[:, :self.dim]
        data.edge_attr = data.edge_attr[:, :self.dim]
        x = self.model.forward(data)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)
 ###############################################################################################
    # Parsing #####################################################################################
    ###############################################################################################
    def parse_output(self, model_output: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nr_bbox = self.num_bounding_boxes

        x_norm_rel = torch.clamp(model_output[..., 0:nr_bbox], min=0)  # Center x
        y_norm_rel = torch.clamp(model_output[..., nr_bbox:nr_bbox*2], min=0)  # Center y
        w_norm_sqrt = torch.clamp(model_output[..., nr_bbox*2:nr_bbox*3], min=0)  # Height
        h_norm_sqrt = torch.clamp(model_output[..., nr_bbox*3:nr_bbox*4], min=0)  # Width
        y_confidence = torch.sigmoid(model_output[..., nr_bbox * 4:nr_bbox * 5])  # Object Confidence
        y_class_scores = model_output[..., nr_bbox * 5:]  # Class Score

        return x_norm_rel, y_norm_rel, w_norm_sqrt, h_norm_sqrt, y_confidence, y_class_scores

    @staticmethod
    def parse_gt(gt_bbox: torch.Tensor, input_shape: torch.Tensor, cell_map_shape: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_corners = yolo_grid(input_shape, cell_map_shape)
        cell_shape = input_shape / cell_map_shape

        gt_cell_corner_offset_x = gt_bbox[..., 0, None] - cell_corners[None, :, 0, 0]
        gt_cell_corner_offset_x[gt_cell_corner_offset_x < 0] = 9999999
        gt_cell_corner_offset_x, gt_cell_x = torch.min(gt_cell_corner_offset_x, dim=-1)

        gt_cell_corner_offset_y = gt_bbox[..., 1, None] - cell_corners[None, 0, :, 1]
        gt_cell_corner_offset_y[gt_cell_corner_offset_y < 0] = 9999999
        gt_cell_corner_offset_y, gt_cell_y = torch.min(gt_cell_corner_offset_y, dim=-1)

        gt_cell_corner_offset = torch.stack([gt_cell_corner_offset_x, gt_cell_corner_offset_y], dim=-1)
        gt_cell_corner_offset_norm = gt_cell_corner_offset / cell_shape[None, :].float()

        gt_bbox_shape = torch.stack([gt_bbox[..., 2], gt_bbox[..., 3]], dim=-1)
        gt_bbox_shape_norm_sqrt = torch.sqrt(gt_bbox_shape / input_shape.float())
        return gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_x, gt_cell_y

    ###############################################################################################
    # YOLO Detection ##############################################################################
    ###############################################################################################
    def detect(self, model_output: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """Computes the detections used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
        cell_map_shape = torch.tensor(model_output.shape[1:3], device=model_output.device)
        input_shape = self.input_shape.to(model_output.device)
        cell_shape = input_shape / cell_map_shape
        x_norm_rel, y_norm_rel, w_norm_sqrt, h_norm_sqrt, pred_conf, pred_cls_conf = self.parse_output(model_output)

        x_rel = x_norm_rel * cell_shape[0]
        y_rel = y_norm_rel * cell_shape[1]
        w = w_norm_sqrt ** 2 * input_shape[0]
        h = h_norm_sqrt**2 * input_shape[1]
        cell_top_left = yolo_grid(input_shape, cell_map_shape)
        bbox_top_left_corner = cell_top_left[None, :, :, None, :] + torch.stack([x_rel, y_rel], dim=-1)

        if threshold is None:
            return torch.cat([bbox_top_left_corner, w.unsqueeze(-1), h.unsqueeze(-1), pred_conf.unsqueeze(-1)], dim=-1)

        detected_bbox_idx = torch.nonzero(torch.gt(pred_conf, threshold)).split(1, dim=-1)
        batch_idx = detected_bbox_idx[0]
        if batch_idx.shape[0] == 0:
            return torch.zeros([0, 7])

        detected_top_left_corner = bbox_top_left_corner[detected_bbox_idx].squeeze(1)
        detected_h = h[detected_bbox_idx]
        detected_w = w[detected_bbox_idx]
        pred_conf = pred_conf[detected_bbox_idx]

        pred_cls = torch.argmax(pred_cls_conf[detected_bbox_idx[:-1]], dim=-1)
        pred_cls_conf = pred_cls_conf[detected_bbox_idx[:-1]].squeeze(1)
        pred_cls_conf = pred_cls_conf[torch.arange(pred_cls.shape[0]), pred_cls.squeeze(-1)]

        # Convert from x, y to u, v
        det_bbox = torch.cat([batch_idx.float(), detected_top_left_corner[:, 0, None].float(),
                             detected_top_left_corner[:, 1, None].float(), detected_w.float(), detected_h.float(),
                             pred_cls.float(), pred_cls_conf[:, None].float(), pred_conf], dim=-1)

        det_bbox[:, 1:5] = crop_to_frame(det_bbox[:, 1:5], image_shape=input_shape)
        return det_bbox

    def detect_nms(self, model_outputs: torch.Tensor, threshold: float = 0.6, nms_iou: float = 0.6):
        detected_bbox = self.detect(model_outputs, threshold=threshold)
        return non_max_suppression(detected_bbox, iou=nms_iou)

