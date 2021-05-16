import numpy as np
import torch
import torch_geometric
import pytorch_lightning as pl
import wandb

from typing import Dict, List
from aegnn.visualize.utils.histogram import compute_histogram


class BBoxLogger(pl.callbacks.base.Callback):

    def on_validation_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if not hasattr(model, "detect"):
            return None

        batch = next(trainer.val_dataloaders[0].__iter__())
        images, p_label, p_bbox, t_label, t_bbox = self.get_bbox(batch, model=model)

        t_class_ids = getattr(batch, "class_id")
        label_unique, indices = torch.unique(t_label, return_indices=True)
        class_id_label_dict = {int(l): t_class_ids[l_idx] for l, l_idx in zip(label_unique, indices)}

        bbox_list = []
        num_images = min(len(images), 10)
        for i in range(num_images):
            wandb_bbox = self.wandb_bbox(images[i], bbox=p_bbox[i].numpy(), label=p_label[i],
                                         true_bbox=t_bbox[i].numpy(), true_label=t_label[i],
                                         class_label_dict=class_id_label_dict)
            bbox_list.append(wandb_bbox)
        wandb.log({"predictions": bbox_list})

    @staticmethod
    def get_bbox(batch: torch_geometric.data.Batch, model: pl.LightningModule):
        img_shape = getattr(model, "img_shape", tuple(batch.pos.max(dim=0).values.numpy().astype(int)))

        with torch.no_grad():
            prediction = model(batch)
            bbox = model.detect(prediction, threshold=0.0)

        images = []
        for i, data in enumerate(batch.to_data_list()):
            hist_image = compute_histogram(data.pos.numpy(), img_shape=img_shape)
            images.append(hist_image)

        p_label, p_bbox = bbox[:, 1:5], bbox[:, 5]
        t_label = batch.y
        t_bbox_np = np.array(getattr(batch, "bb")).reshape(-1, 5)
        t_bbox = torch.from_numpy(t_bbox_np)[:, :4]
        return images, p_label, p_bbox, t_label, t_bbox

    @staticmethod
    def wandb_bbox(image, bbox: List[int], label: int, true_bbox: List[int], true_label: int,
                   class_label_dict: Dict[int, str]) -> wandb.Image:
        return wandb.Image(image, boxes={
            "predictions": {
                "box_data": [{
                    "position": {
                        "minX": bbox[0],
                        "maxX": bbox[2],
                        "minY": bbox[1],
                        "maxY": bbox[3]
                    },
                    "class_id": label,
                    "box_caption": class_label_dict[label]
                }],
                "class_labels": class_label_dict
            },
            "ground_truth": {
                "box_data": [{
                    "position": {
                        "minX": true_bbox[0],
                        "maxX": true_bbox[2],
                        "minY": true_bbox[1],
                        "maxY": true_bbox[3]
                    },
                    "class_id": true_label,
                    "box_caption": class_label_dict[true_label]
                }],
                "class_labels": class_label_dict
            }
        })
