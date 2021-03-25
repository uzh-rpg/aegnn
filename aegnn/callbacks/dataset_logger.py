import collections
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm


class DatasetLogger(pl.callbacks.base.Callback):

    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        print("Logging dataset information")
        log_info = collections.defaultdict(list)

        # Determine the samples taken from each dataset. Depending on the size of the dataset,
        # the number of samples is either limited by 100 or the size of the dataset.
        ds = getattr(trainer.train_dataloader.dataset, "datasets")  # -> pl.trainer.supporters.CombinedDataset
        num_samples = min(100, len(ds))
        samples = np.random.randint(0, len(ds), size=num_samples)  # high = len(ds) is exclusive(!)

        # Iterate over training batches while logging graph properties for every graph in each batch.
        for i in tqdm(samples):
            data = ds.get(i)
            log_info["dataset.num_nodes"] = data.num_nodes
            log_info["dataset.num_edges"] = data.num_edges

        # Log the average value over each graph property, average being the estimated mean of each
        # property over all graphs.
        log_info = {key: np.mean(values) for key, values in log_info.items()}
        model.logger.log_hyperparams(log_info)
