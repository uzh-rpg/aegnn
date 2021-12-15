import aegnn
import argparse
import itertools
import logging
import os
import numpy as np
import pandas as pd
import torch

from aegnn.models.utils.map import compute_map
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path of model to evaluate.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()


def main(model, data_module, num_workers: int = 16):
    results_df = pd.DataFrame()

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"map_search_{args.dataset}.pkl")

    iou_thresholds_range = np.linspace(0, 1.0, num=11)
    confidence_thresholds = np.linspace(0.0, 0.8, num=4)
    nms_thresholds = np.linspace(0.3, 0.80, num=4)
    thresholds = list(itertools.product(confidence_thresholds, nms_thresholds))

    for confidence_th, nms_th in tqdm(thresholds):
        logging.debug(f"Evaluating mAP for confidence = {confidence_th} and nms = {nms_th}")
        map_scores = []

        with torch.no_grad():
            data_loader = data_module.val_dataloader(num_workers=num_workers).__iter__()
            for batch in tqdm(data_loader):
                batch = batch.to(model.device)
                outputs = model.forward(data=batch)
                detected_bbox = model.detect_nms(outputs, threshold=confidence_th, nms_iou=nms_th)

                gt_batch = getattr(batch, "batch_bbox").detach()
                gt_bb = getattr(batch, "bbox")
                map_scores.append(compute_map(detected_bbox, gt_bb, gt_batch, iou_thresholds_range))

        results_dict = dict(nms=nms_th, confidence=confidence_th)
        results_dict["mAP"] = np.mean(map_scores)
        logging.debug(f"Results: {results_dict}")
        results_df = results_df.append(results_dict, ignore_index=True)
        results_df.to_pickle(output_file)

    print(f"Results are logged in {output_file}")
    return results_df


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    if args.device != "cpu":
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    model_eval = torch.load(args.model_file).to(device)
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()
    main(model_eval, data_module=dm)
