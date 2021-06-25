import numpy as np
import torch
import torch_geometric
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb

from typing import Any, Dict, List
from aegnn.utils.bounding_box import non_max_suppression
from aegnn.visualize.utils.histogram import compute_histogram


class BBoxLogger(pl.callbacks.base.Callback):

    def __init__(self, classes: List[str], max_num_images: int = 8, max_num_bbox: int = 3, padding: int = 50):
        self.classes = np.array(classes)
        self.__max_num_images = max_num_images
        self.__max_num_bbox = max_num_bbox
        self.__padding = padding

        self.__batch_w_outputs = None

    def on_validation_batch_end(self, trainer, model: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.__batch_w_outputs is None:
            batch.outputs = outputs.detach()
            self.__batch_w_outputs = batch
        return None

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if not hasattr(model, "detect") or self.__batch_w_outputs is None:
            return None
        images, p_bboxes, t_bboxes = self.get_bbox(self.__batch_w_outputs, model=model)
        t_bbox_batch = getattr(self.__batch_w_outputs, "batch_bbox").detach().cpu().numpy()
        self.__batch_w_outputs = None

        # Bring the bounding boxes (prediction & ground-truth) to a json-like format to be
        # understandable for the logger object. For the sake of training speed, limit the number
        # of images to some predefined upper limit.
        num_images = min(len(images), self.__max_num_images)
        boxes_formatted = []
        for i in range(num_images):
            p_bbox = p_bboxes[p_bboxes[:, 0] == i, 1:]
            p_bbox = p_bbox[p_bbox[:, -1].argsort(), :][-self.__max_num_bbox:, :]   # n highest confidence scores
            t_bbox = t_bboxes[t_bbox_batch == i, :]
            wandb_bbox = self.wandb_bbox(p_bboxes=p_bbox, t_bboxes=t_bbox, padding=self.__padding)
            boxes_formatted.append(wandb_bbox)

        # If the model logger is a WandB Logger, convert the image and bounding boxes to the WandB format
        # and log them using its API (i.e. upload the images).
        if isinstance(model.logger, pytorch_lightning.loggers.WandbLogger):
            wandb_data = []
            for i in range(num_images):
                image = np.pad(images[i], pad_width=self.__padding)
                wandb_data.append(wandb.Image(image, boxes=boxes_formatted[i]))
            model.logger.experiment.log({"predictions": wandb_data}, commit=False)

    @staticmethod
    def get_bbox(batch: torch_geometric.data.Batch, model: pl.LightningModule):
        """Get bounding boxes, both predicted and ground-truth bounding boxes, from batch and compute the
        underlying 2D representation of the events by stacking the events in pixel-wise buckets."""
        img_shape = getattr(model, "input_shape", None)

        with torch.no_grad():
            bbox = model.detect(getattr(batch, "outputs"), threshold=0.3)
            bbox = non_max_suppression(bbox, iou=0.6)

        images = []
        for i, data in enumerate(batch.to_data_list()):
            hist_image = compute_histogram(data.pos.cpu().numpy(), img_shape=img_shape, max_count=1)
            images.append(hist_image.T)

        p_bbox_np = bbox.detach().cpu().numpy()
        t_bbox_np = getattr(batch, "bbox").cpu().numpy().reshape(-1, 5)
        return images, p_bbox_np, t_bbox_np

    ###############################################################################################
    # Write W&B Bounding Boxes ####################################################################
    ###############################################################################################
    def __write_bbox(self, bounding_boxes: np.ndarray, padding: int):
        bbox_data = []
        for bbox in bounding_boxes:
            label = self.classes[int(bbox[4])]
            bbox_dict = {
                "position": {
                    "minX": int(bbox[0] + padding),
                    "maxX": int(bbox[0] + padding + bbox[2]),
                    "minY": int(bbox[1] + padding),
                    "maxY": int(bbox[1] + padding + bbox[3])
                },
                "class_id": int(bbox[4]),
                "box_caption": label,
                "domain": "pixel"
            }
            bbox_data.append(bbox_dict)
        return bbox_data

    def wandb_bbox(self, p_bboxes: np.ndarray, t_bboxes: np.ndarray, padding: int) -> Dict:
        class_id_label_dict = {i: class_id for i, class_id in enumerate(self.classes)}
        p_bbox_data = self.__write_bbox(p_bboxes, padding=padding)
        t_bbox_data = self.__write_bbox(t_bboxes, padding=padding)

        boxes = {
            "predictions": {
                "box_data": p_bbox_data,
                "class_labels": class_id_label_dict
            },
            "ground_truth": {
                "box_data": t_bbox_data,
                "class_labels": class_id_label_dict
            },
        }
        return boxes
