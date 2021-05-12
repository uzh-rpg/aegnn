"""Partly copied from rpg-asynet paper: https://github.com/uzh-rpg/rpg_asynet"""
import torch
import torch_geometric
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as pl_metrics

from typing import Dict, Tuple

from ..utils import bounding_box as model_utils
from ..utils import metrics as model_metrics
from ..utils.pascalvoc import BBType, BoundingBox, BoundingBoxes, MethodAveragePrecision, VOC_Evaluator


class DetectionModel(pl.LightningModule):

    __lambda_coord = 5
    __lambda_no_object = 0.5
    __lambda_class = 1

    def __init__(self, num_classes: int, num_bounding_boxes: int = 1, learning_rate: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.num_bounding_boxes = num_bounding_boxes
        # self.cell_size = (5, 7)
        self.cell_map_shape = (1, 1)
        self.input_shape = torch.tensor((255, 191))  # ncaltech101!!

        self.num_outputs_per_cell = num_classes + num_bounding_boxes * 5  # (x, y, width, height, confidence)
        self.num_outputs = self.num_outputs_per_cell * self.cell_map_shape[0] * self.cell_map_shape[1]  # detection grid

        self.optimizer_kwargs = dict(lr=learning_rate)
        self.evaluator = VOC_Evaluator()

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)

        # Compute loss, as weighted sum of multiple loss functions.
        gt_bb = torch.tensor(getattr(batch, "bb"), device=self.device)
        loss, losses_dict, iou = self.loss(outputs, bounding_box=gt_bb)
        loss_logs = {f"Train/Loss-{name.capitalize()}": value for name, value in losses_dict.items()}

        # Compute metrics for the recognition (class accuracy) and detection (iou) part, as well as combined
        # metrics (mean average precision = mAP).
        metrics_logs = self.evaluate(outputs, batch=batch, prefix="Train/")
        metrics_logs["Train/IOU"] = iou.mean()
        self.logger.log_metrics({"Train/Loss": loss, **loss_logs, **metrics_logs}, step=self.trainer.global_step)
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        gt_bb = torch.tensor(getattr(batch, "bb"), device=self.device)

        loss, _, iou = self.loss(outputs, bounding_box=gt_bb)
        metrics_logs = self.evaluate(outputs, batch=batch, prefix="Val/")
        metrics_logs["Val/IOU"] = iou.mean()
        self.logger.log_metrics({"Val/Loss": loss, **metrics_logs}, step=self.trainer.global_step)

        return outputs

    ###############################################################################################
    # Parsing #####################################################################################
    ###############################################################################################
    def parse_output(self, model_output: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nr_bbox = self.num_bounding_boxes

        x_norm_rel = model_output[..., 0:nr_bbox]  # Center x
        y_norm_rel = model_output[..., nr_bbox:nr_bbox*2]  # Center y
        h_norm_sqrt = model_output[..., nr_bbox*2:nr_bbox*3]  # Height
        w_norm_sqrt = model_output[..., nr_bbox*3:nr_bbox*4]  # Width
        y_confidence = torch.sigmoid(model_output[..., nr_bbox * 4:nr_bbox * 5])  # Object Confidence
        y_classes = model_output[..., nr_bbox * 5:]  # Class Score

        return x_norm_rel, y_norm_rel, h_norm_sqrt, w_norm_sqrt, y_confidence, y_classes

    @staticmethod
    def parse_gt(bounding_box: torch.Tensor, input_shape: torch.Tensor, cell_map_shape: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalizes and computes the offset to the grid corner"""
        # Construct normalized, relative ground truth
        cell_corners = model_utils.get_grid(input_shape, cell_map_shape)
        cell_shape = input_shape / cell_map_shape
        # bounding_box[0, 0, :]: ['u', 'v', 'w', 'h', 'class_id'].  (u, v) is top left point
        # gt_bbox_center = bounding_box[:, :, :2] + bounding_box[:, :, 2:4] // 2
        gt_bbox_center = bounding_box[:, :, :2]
        # (u, v) -> (x, y)
        gt_cell_corner_offset_x = gt_bbox_center[:, :, 0, None] - cell_corners[None, None, :, 0, 1]
        gt_cell_corner_offset_x[gt_cell_corner_offset_x < 0] = 0
        gt_cell_corner_offset_x, gt_cell_x = torch.min(gt_cell_corner_offset_x, dim=-1)

        gt_cell_corner_offset_y = gt_bbox_center[:, :, 1, None] - cell_corners[None, None, 0, :, 0]
        gt_cell_corner_offset_y[gt_cell_corner_offset_y < 0] = 0
        gt_cell_corner_offset_y, gt_cell_y = torch.min(gt_cell_corner_offset_y, dim=-1)

        gt_cell_corner_offset = torch.stack([gt_cell_corner_offset_x, gt_cell_corner_offset_y], dim=-1)
        gt_cell_corner_offset_norm = gt_cell_corner_offset / cell_shape[None, None, :].float()

        # (width, height) -> (height, width)
        gt_bbox_shape = torch.stack([bounding_box[:, :, 3], bounding_box[:, :, 2]], dim=-1)
        gt_bbox_shape_norm_sqrt = torch.sqrt(gt_bbox_shape / input_shape.float())
        return gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_y, gt_cell_x

    ###############################################################################################
    # Metrics #####################################################################################
    ###############################################################################################
    def evaluate(self, model_outputs: torch.Tensor, batch: torch_geometric.data.Batch, prefix: str = ""
                 ) -> Dict[str, torch.Tensor]:
        metrics_logs = {}

        detected_bb = self.detect(model_outputs, threshold=0.0)
        dbb_batch_idx = detected_bb[:, 0].long()
        metrics_logs[f"{prefix}Accuracy"] = pl_metrics.accuracy(detected_bb[:, 5].long(), target=batch.y[dbb_batch_idx])

        with torch.no_grad():
            gt_bb = torch.tensor(getattr(batch, "bb"), device=self.device)
            detected_bb = self.detect(model_outputs, threshold=0.3)
            detected_bb = model_utils.non_max_suppression(detected_bb, iou=0.6)
            detected_bb = detected_bb.detach().cpu().numpy()

            training_bbs = self.add_pascalvoc_bounding_boxes(BoundingBoxes(), gt_bb, detected_bbox=detected_bb,
                                                             image_id=getattr(batch, "file_id"))
            map_method = MethodAveragePrecision.EveryPointInterpolation
            metrics = self.evaluator.GetPascalVOCMetrics(training_bbs, IOUThreshold=0.5, method=map_method)
            metrics_logs[f"{prefix}mAP"] = sum([m["AP"] for m in metrics]) / self.num_classes

        return metrics_logs

    def add_pascalvoc_bounding_boxes(self, bounding_boxes: BoundingBoxes, gt_bbox: torch.Tensor,
                                     detected_bbox: torch.Tensor, image_id: str) -> BoundingBoxes:
        """Saves the bounding boxes in the evaluation format
        :param bounding_boxes: bounding boxes to add pascal formatted bbox to.
        :param gt_bbox: gt_bbox: ['u', 'v', 'w', 'h', 'class_id']
        :param detected_bbox: [batch_idx, u, v, w, h, pred_class_id, pred_class_score, object score]
        :param image_id: image identifier.
        """
        image_size = self.input_shape.detach().cpu().numpy()
        for i_batch in range(gt_bbox.shape[0]):
            for i_gt in range(gt_bbox.shape[1]):
                gt_bbox_sample = gt_bbox[i_batch, i_gt, :]
                if gt_bbox[i_batch, i_gt, :].sum() == 0:
                    break

                bb_gt = BoundingBox(image_id[i_batch], gt_bbox_sample[-1], gt_bbox_sample[0], gt_bbox_sample[1],
                                    gt_bbox_sample[2], gt_bbox_sample[3], image_size, BBType.GroundTruth)
                bounding_boxes.addBoundingBox(bb_gt)

        for i_det in range(detected_bbox.shape[0]):
            det_bbox_sample = detected_bbox[i_det, :]
            i_batch = int(det_bbox_sample[0])
            bb_det = BoundingBox(image_id[i_batch], det_bbox_sample[5], det_bbox_sample[1], det_bbox_sample[2],
                                 det_bbox_sample[3], det_bbox_sample[4], image_size, BBType.Detected,
                                 det_bbox_sample[6])
            bounding_boxes.addBoundingBox(bb_det)

        return bounding_boxes

    ###############################################################################################
    # YOLO Detection ##############################################################################
    ###############################################################################################
    def detect(self, model_output: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """Computes the detections used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
        cell_map_shape = torch.tensor(model_output.shape[1:3], device=model_output.device)
        input_shape = self.input_shape.to(model_output.device)
        cell_shape = input_shape / cell_map_shape
        x_norm_rel, y_norm_rel, h_norm_sqrt, w_norm_sqrt, pred_conf, pred_cls_conf = self.parse_output(model_output)

        h = h_norm_sqrt**2 * input_shape[0]
        w = w_norm_sqrt**2 * input_shape[1]

        x_rel = x_norm_rel * cell_shape[0]
        y_rel = y_norm_rel * cell_shape[1]
        cell_top_left = model_utils.get_grid(input_shape, cell_map_shape)
        bbox_top_left_corner = cell_top_left[None, :, :, None, :] + torch.stack([x_rel, y_rel], dim=-1)

        if threshold is None:
            return torch.cat([bbox_top_left_corner, h.unsqueeze(-1), w.unsqueeze(-1), pred_conf.unsqueeze(-1)], dim=-1)

        detected_bbox_idx = torch.gt(pred_conf, threshold).nonzero().split(1, dim=-1)
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
        det_bbox = torch.cat([batch_idx.float(), detected_top_left_corner[:, 1, None].float(),
                             detected_top_left_corner[:, 0, None].float(), detected_w.float(), detected_h.float(),
                             pred_cls.float(), pred_cls_conf[:, None].float(), pred_conf], dim=-1)

        return model_utils.crop_to_frame(det_bbox, input_shape)

    ###############################################################################################
    # YOLO Loss ###################################################################################
    ###############################################################################################
    def loss(self, model_output: torch.Tensor, bounding_box: torch.Tensor
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Computes the loss used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
        batch_s, gt_nr_bbox, _ = bounding_box.shape
        input_shape = self.input_shape.to(model_output.device)
        cell_map_shape = torch.tensor(model_output.shape[1:3], device=model_output.device)
        valid_gt_bbox_mask = bounding_box.sum(dim=-1) > 0

        x_offset_norm, y_offset_norm, h_norm_sqrt, w_norm_sqrt, pred_conf, pred_cls = self.parse_output(model_output)
        out = self.parse_gt(bounding_box, input_shape, cell_map_shape)
        gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_x, gt_cell_y = out

        # Get IoU at gt_bbox_position
        bbox_detection = self.detect(model_output, threshold=None)
        batch_indices = torch.arange(batch_s)[:, None].repeat([1, gt_nr_bbox])
        bbox_detection = bbox_detection[batch_indices, gt_cell_x, gt_cell_y, :, :]
        iou = model_metrics.compute_iou(bbox_detection[:, :, :, :4], gt_bbox=bounding_box[:, :, :4]).detach()
        confidence_score, responsible_pred_bbox_idx = torch.max(iou, dim=-1)

        # ----- Offset Loss -----
        pred_cell_offset_norm = torch.stack([x_offset_norm, y_offset_norm], dim=-1)
        # Get the predictions, which include a object and correspond to the responsible cell
        pred_cell_offset_norm = pred_cell_offset_norm[batch_indices, gt_cell_x, gt_cell_y, responsible_pred_bbox_idx, :]

        offset_delta = pred_cell_offset_norm - gt_cell_corner_offset_norm
        offset_loss = (offset_delta ** 2).sum(-1)[valid_gt_bbox_mask].mean()

        # ----- Height & Width Loss -----
        pred_cell_shape_norm_sqrt = torch.stack([h_norm_sqrt, w_norm_sqrt], dim=-1)
        # Get the predictions, which include a object and correspond to the responsible cell
        pred_cell_shape_norm_sqrt = pred_cell_shape_norm_sqrt[batch_indices, gt_cell_x, gt_cell_y,
                                                              responsible_pred_bbox_idx, :]
        shape_delta = pred_cell_shape_norm_sqrt - gt_bbox_shape_norm_sqrt
        shape_loss = (shape_delta ** 2).sum(-1)[valid_gt_bbox_mask].mean()

        # ----- Object Confidence Loss -----
        # Get the predictions, which include a object and correspond to the responsible cell
        pred_conf_object = pred_conf[batch_indices, gt_cell_x, gt_cell_y, responsible_pred_bbox_idx]
        confidence_delta = pred_conf_object - confidence_score
        confidence_loss = (confidence_delta ** 2)[valid_gt_bbox_mask].mean()

        # ----- No Object Confidence Loss -----
        # Get the predictions, which do not include a object
        no_object_mask = torch.ones_like(pred_conf)
        no_object_mask[batch_indices[valid_gt_bbox_mask], gt_cell_x[valid_gt_bbox_mask], gt_cell_y[valid_gt_bbox_mask],
                       responsible_pred_bbox_idx[valid_gt_bbox_mask]] = 0

        if torch.any(no_object_mask.bool()):
            confidence_no_object_loss = (pred_conf[no_object_mask.bool()] ** 2).mean()
        else:
            confidence_no_object_loss = torch.tensor(0, device=model_output.device)

        # ----- Class Prediction Loss -----
        loss_function = torch.nn.CrossEntropyLoss()
        pred_class_bbox = pred_cls[batch_indices, gt_cell_x, gt_cell_y, :][valid_gt_bbox_mask]
        class_label = bounding_box[:, :, -1][valid_gt_bbox_mask]
        class_loss = loss_function(pred_class_bbox, target=class_label)

        losses_dict = {"offset": self.__lambda_coord * offset_loss,
                       "shape": self.__lambda_coord * shape_loss,
                       "confidence": confidence_loss,
                       "confidence_no_object": self.__lambda_no_object * confidence_no_object_loss,
                       "class": self.__lambda_class * class_loss}
        return sum(losses_dict.values()), losses_dict, iou
