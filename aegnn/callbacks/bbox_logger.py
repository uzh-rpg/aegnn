import numpy as np
import torch
import torch_geometric
import pytorch_lightning as pl
import wandb

from typing import List
from aegnn.visualize.utils.histogram import compute_histogram


class BBoxLogger(pl.callbacks.base.Callback):

    def __init__(self, classes: List[str], max_num_images: int = 8):
        self.classes = np.array(classes)
        self._max_num_images = max_num_images

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if not hasattr(model, "detect"):
            return None

        batch = next(model.train_dataloader().__iter__()).to(model.device)
        images, p_label, p_bbox, t_label, t_bbox = self.get_bbox(batch, model=model)
        p_bbox, p_label = p_bbox.cpu().numpy(), p_label.cpu().numpy()
        t_bbox, t_label = t_bbox.cpu().numpy(), t_label.cpu().numpy()

        bbox_list = []
        num_images = min(len(images), self._max_num_images)
        for i in range(num_images):
            wandb_bbox = self.wandb_bbox(images[i], bbox=p_bbox[i].tolist(), label=p_label[i],
                                         t_bbox=t_bbox[i].tolist(), t_label=t_label[i])
            bbox_list.append(wandb_bbox)
        if hasattr(model.logger.experiment, "log"):
            model.logger.experiment.log({"predictions": bbox_list}, step=trainer.global_step, commit=False)

    @staticmethod
    def get_bbox(batch: torch_geometric.data.Batch, model: pl.LightningModule):
        img_shape = getattr(model, "input_shape", None)

        with torch.no_grad():
            prediction = model.forward(batch)
            bbox = model.detect(prediction, threshold=0.0)

        images = []
        for i, data in enumerate(batch.to_data_list()):
            hist_image = compute_histogram(data.pos.cpu().numpy(), img_shape=img_shape, max_count=1)
            images.append(hist_image.T)

        p_bbox, p_label = bbox[:, 1:5], bbox[:, 5]
        t_label = batch.y
        t_bbox_np = getattr(batch, "bbox").cpu().numpy().reshape(-1, 5)
        t_bbox = torch.from_numpy(t_bbox_np)[:, :4]
        return images, p_label, p_bbox, t_label, t_bbox

    def wandb_bbox(self, image: np.ndarray, bbox: List[int], label: int, t_bbox: List[int], t_label: int,
                   padding: int = 50) -> wandb.Image:
        p_class_id = self.classes[int(label)]
        t_class_id = self.classes[int(t_label)]
        class_id_label_dict = {i: class_id for i, class_id in enumerate(self.classes)}

        image = np.pad(image, pad_width=padding)
        return wandb.Image(image, boxes={
            "predictions": {
                "box_data": [{
                    "position": {
                        "minX": bbox[0] + padding,
                        "maxX": bbox[0] + padding + bbox[2],
                        "minY": bbox[1] + padding,
                        "maxY": bbox[1] + padding + bbox[3]
                    },
                    "class_id": int(label),
                    "box_caption": p_class_id,
                    "domain": "pixel"
                }],
                "class_labels": class_id_label_dict
            },
            "ground_truth": {
                "box_data": [{
                    "position": {
                        "minX": t_bbox[0] + padding,
                        "maxX": t_bbox[0] + padding + t_bbox[2],
                        "minY": t_bbox[1] + padding,
                        "maxY": t_bbox[1] + padding + t_bbox[3],
                    },
                    "class_id": int(t_label),
                    "box_caption": t_class_id,
                    "domain": "pixel"
                }],
                "class_labels": class_id_label_dict
            },
        })
