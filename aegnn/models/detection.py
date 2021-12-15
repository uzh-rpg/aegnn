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

    __lambda_coord = 2
    __lambda_no_object = 0.5
    __lambda_class = 1

    def __init__(self, network: str, dataset: str, num_classes: int, img_shape: Tuple[int, int], dim: int = 3,
                 num_bounding_boxes: int = 1, learning_rate: float = 1e-3, **model_kwargs):
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

        # Additional arguments for optimization and logging.
        self.optimizer_kwargs = dict(lr=learning_rate)
        self.__validation_logs = collections.defaultdict(list)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        data.pos = data.pos[:, :self.dim]
        data.edge_attr = data.edge_attr[:, :self.dim]
        x = self.model.forward(data)
        return x.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch.clone())

        # Compute loss, as weighted sum of multiple loss functions.
        gt_bb = getattr(batch, "bbox").to(self.device)
        gt_bb_batch = getattr(batch, "batch_bbox")
        logging.debug(f"Loss computation on batch with {len(gt_bb)} ground-truth bounding boxes")
        loss, losses_dict, iou = self.loss(outputs, bounding_box=gt_bb, bbox_batch=gt_bb_batch)
        loss_logs = {f"Train/Loss-{name.capitalize()}": value for name, value in losses_dict.items()}

        # Compute metrics for the recognition (class accuracy) and detection (iou) part, as well as combined
        # metrics (mean average precision = mAP).
        logging.debug("Parsing model outputs to evaluate the model performance")
        gt_batch = gt_bb_batch.detach().cpu()
        detected_bbox = self.detect_nms(model_outputs=outputs)

        train_accuracy = compute_detection_accuracy(detected_bbox, gt_y=batch.y.detach().cpu(), gt_batch=gt_batch)
        train_map = compute_map(detected_bbox, gt_bbox=gt_bb.detach().cpu(), gt_batch=gt_batch)
        metrics_logs = {"Train/Accuracy": train_accuracy, "Train/mAP": train_map}

        # Send loss and evaluation metrics to logger for logging.
        self.logger.log_metrics({"Train/Loss": loss, "Train/IOU": iou.mean(), **loss_logs, **metrics_logs})
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        detected_bbox = self.detect_nms(model_outputs=outputs)
        gt_bb = getattr(batch, "bbox").to(self.device)
        gt_bb_batch = getattr(batch, "batch_bbox")

        # Compute evaluation metrics for the current batch (recognition accuracy, mAP score).
        with torch.no_grad():
            gt_batch = gt_bb_batch.detach().cpu()
            loss, _, iou = self.loss(outputs, bounding_box=gt_bb, bbox_batch=gt_bb_batch)
            val_accuracy = compute_detection_accuracy(detected_bbox, gt_y=batch.y.detach().cpu(), gt_batch=gt_batch)
            val_map = compute_map(detected_bbox, gt_bbox=gt_bb.detach().cpu(), gt_batch=gt_batch)

        # Append to the validation log dictionary for accumulated logging on validation end.
        self.__validation_logs["Loss"].append(loss.detach().cpu().item())
        self.__validation_logs["IOU"].append(iou.detach().cpu().item())
        self.__validation_logs["Accuracy"].append(val_accuracy)
        self.__validation_logs["mAP"].append(val_map)
        return outputs

    def on_validation_end(self) -> None:
        metrics_logs = {f"Val/{key}": np.mean(values) for key, values in self.__validation_logs.items()}
        self.logger.log_metrics(metrics_logs)
        self.__validation_logs = collections.defaultdict(list)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4, **self.optimizer_kwargs)

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

    ###############################################################################################
    # YOLO Loss ###################################################################################
    ###############################################################################################
    def loss(self, model_output: torch.Tensor, bounding_box: torch.Tensor, bbox_batch: torch.LongTensor
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Computes the loss used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
        input_shape = self.input_shape.to(model_output.device)
        cell_map_shape = torch.tensor(model_output.shape[1:3], device=model_output.device)

        ious = []
        loss_keys = ["offset", "shape", "confidence", "confidence_no_object", "class"]
        losses_dict = {key: torch.zeros(1, device=model_output.device) for key in loss_keys}
        for batch_i in bbox_batch.long().unique():
            gt_bbox_i = bounding_box[bbox_batch == batch_i, :]

            x_offset_norm, y_offset_norm, w_norm_sqrt, h_norm_sqrt, pred_conf, pred_cls = self.parse_output(
                model_output[batch_i])
            out = self.parse_gt(gt_bbox_i, input_shape, cell_map_shape)
            gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_x, gt_cell_y = out

            # Get IoU at gt_bbox_position
            bbox_detection = self.detect(model_output, threshold=None)
            bbox_detection = bbox_detection[batch_i, gt_cell_x, gt_cell_y, :, :]
            iou = compute_iou(bbox_detection[..., :4], gt_bbox=gt_bbox_i[..., :4])
            confidence_score, responsible_pred_bbox_idx = torch.max(iou, dim=-1)

            # ----- Offset Loss -----
            # Get the predictions, which include a object and correspond to the responsible cell
            pred_cell_offset_norm = torch.stack([x_offset_norm, y_offset_norm], dim=-1)
            pred_cell_offset_norm = pred_cell_offset_norm[gt_cell_x, gt_cell_y, responsible_pred_bbox_idx, :]
            offset_delta = pred_cell_offset_norm - gt_cell_corner_offset_norm
            offset_loss = (offset_delta ** 2).sum(-1).mean()

            # ----- Height & Width Loss -----
            # Get the predictions, which include a object and correspond to the responsible cell
            pred_cell_shape_norm_sqrt = torch.stack([w_norm_sqrt, h_norm_sqrt], dim=-1)
            pred_cell_shape_norm_sqrt = pred_cell_shape_norm_sqrt[gt_cell_x, gt_cell_y, responsible_pred_bbox_idx, :]
            shape_delta = pred_cell_shape_norm_sqrt - gt_bbox_shape_norm_sqrt
            shape_loss = (shape_delta ** 2).sum(-1).mean()

            # ----- Object Confidence Loss -----
            # Get the predictions, which include a object and correspond to the responsible cell
            pred_conf_object = pred_conf[gt_cell_x, gt_cell_y, responsible_pred_bbox_idx]
            confidence_delta = pred_conf_object - confidence_score
            confidence_loss = (confidence_delta ** 2).mean()

            # ----- No Object Confidence Loss -----
            # Get the predictions, which do not include a object
            no_object_mask = torch.ones_like(pred_conf, dtype=torch.bool)
            no_object_mask[gt_cell_x, gt_cell_y, responsible_pred_bbox_idx] = 0
            if torch.any(no_object_mask):
                confidence_no_object_loss = (pred_conf[no_object_mask] ** 2).mean()
            else:
                confidence_no_object_loss = torch.tensor(0, device=model_output.device)

            # ----- Class Prediction Loss -----
            loss_function = torch.nn.CrossEntropyLoss()
            pred_class_bbox = pred_cls[gt_cell_x, gt_cell_y, :]
            class_label = gt_bbox_i[:, -1]
            class_loss = loss_function(pred_class_bbox, target=class_label)

            ious.append(iou)
            losses_dict["offset"] += self.__lambda_coord * offset_loss
            losses_dict["shape"] += self.__lambda_coord * shape_loss
            losses_dict["confidence"] += confidence_loss
            losses_dict["confidence_no_object"] += self.__lambda_no_object * confidence_no_object_loss
            losses_dict["class"] += self.__lambda_class * class_loss

        # Compute IOU mean value. When there are no bounding boxes, assume an IOU = 1.
        iou_mean = torch.cat(ious).mean() if len(ious) > 0 else torch.ones(1, device=model_output.device)
        return sum(losses_dict.values()), losses_dict, iou_mean
