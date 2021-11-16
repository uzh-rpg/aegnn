import logging

import numpy as np
import torch
import torch_geometric
import pytorch_lightning as pl
import pytorch_lightning.loggers
import wandb

from typing import Any, Dict, List
from wandb.wandb_run import Run as WandBRun
from aegnn.visualize.utils.histogram import compute_histogram


class BBoxLogger(pl.callbacks.base.Callback):

    def __init__(self, classes: List[str], max_num_images: int = 4, padding: int = 50):
        self.classes = np.array(classes)
        self.__max_num_images = max_num_images
        self.__padding = padding

        self.__batch_w_outputs = None

    def on_validation_batch_end(self, trainer, model, outputs: Any, batch, batch_idx: int, dataloader_idx: int) -> None:
        if batch_idx > 0:
            return
        logging.debug("Current model outputs cached for bounding box logging")
        batch.outputs = outputs.clone()
        self.__batch_w_outputs = batch

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
            t_bbox = t_bboxes[t_bbox_batch == i, :]
            wandb_bbox = self.wandb_bbox(p_bboxes=p_bbox, t_bboxes=t_bbox, padding=self.__padding)
            boxes_formatted.append(wandb_bbox)

        # If the model logger is a WandB Logger, convert the image and bounding boxes to the WandB format
        # and log them using its API (i.e. upload the images).
        if isinstance(model.logger, pytorch_lightning.loggers.WandbLogger):
            logging.debug("Logger of type WandbLogger detected, sending bbox images to it")
            exp = model.logger.experiment
        elif isinstance(model.logger, pytorch_lightning.loggers.LoggerCollection):
            logging.debug("Logger of type LoggerCollection detected, searching for WandBLogger types")
            exps = list(filter(lambda x: isinstance(x, WandBRun), model.logger.experiment))
            exp = exps[0] if len(exps) > 0 else None
        else:
            exp = None
        self.send_to_wandb(exp, images, boxes_formatted, padding=self.__padding)

    @staticmethod
    def get_bbox(batch: torch_geometric.data.Batch, model: pl.LightningModule):
        """Get bounding boxes, both predicted and ground-truth bounding boxes, from batch and compute the
        underlying 2D representation of the events by stacking the events in pixel-wise buckets."""
        img_shape = getattr(model, "input_shape", None)

        with torch.no_grad():
            bbox = model.detect_nms(getattr(batch, "outputs"), threshold=0.01)

        images = []
        for i, data in enumerate(batch.to_data_list()):
            hist_image = compute_histogram(data.pos.cpu().numpy(), img_shape=img_shape, max_count=1)
            images.append(hist_image.T)

        p_bbox_np = bbox.detach().cpu().numpy()
        t_bbox_np = getattr(batch, "bbox").cpu().numpy().reshape(-1, 5)
        return images, p_bbox_np, t_bbox_np

    ###############################################################################################
    # I/O WandB ###################################################################################
    ###############################################################################################
    @staticmethod
    def send_to_wandb(experiment, images: List[np.ndarray], boxes: List[dict], padding: int = 0):
        if experiment is None:
            return
        wandb_data = []
        for i, bbox_i in enumerate(boxes):
            image = np.pad(images[i], pad_width=padding)
            wandb_data.append(wandb.Image(image, boxes=bbox_i))
        experiment.log({"predictions": wandb_data}, commit=False)

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
                "domain": "pixel",
            }

            if len(bbox) > 5:
                confidence = float(bbox[6])
                bbox_dict["scores"] = {"class_confidence": float(bbox[5]),
                                       "confidence": confidence}
                bbox_dict["box_caption"] = f"{label}({confidence:.2f})"
            bbox_data.append(bbox_dict)
        return bbox_data

    def wandb_bbox(self, p_bboxes: np.ndarray, t_bboxes: np.ndarray, padding: int) -> Dict:
        class_id_label_dict = {i: class_id for i, class_id in enumerate(self.classes)}
        p_bbox_data = self.__write_bbox(p_bboxes, padding=padding)
        t_bbox_data = self.__write_bbox(t_bboxes, padding=padding)
        logging.debug(f"Added {len(p_bbox_data)} prediction and {len(t_bbox_data)} gt bounding boxes")

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
