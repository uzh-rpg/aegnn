import aegnn
import argparse
import itertools
import logging
import os
import pandas as pd
import torch
import torch_geometric

from torch_geometric.data import Data
from tqdm import tqdm
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(10, 10))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


##################################################################################################
# Graph Generation ###############################################################################
##################################################################################################
def sample_initial_data(sample, num_events: int, radius: float, edge_attr, max_num_neighbors: int):
    data = Data(x=sample.x[:num_events], pos=sample.pos[:num_events])
    data.batch = torch.zeros(data.num_nodes, device=data.x.device)
    data.edge_index = torch_geometric.nn.radius_graph(data.pos, r=radius, max_num_neighbors=max_num_neighbors).long()
    data.edge_attr = edge_attr(data).edge_attr

    edge_counts_avg = data.edge_index.shape[1] / num_events
    logging.debug(f"Average edge counts in initial data = {edge_counts_avg}")
    return data


def create_and_run_model(dm, num_events: int, index: int, device: torch.device, args: argparse.Namespace, **kwargs):
    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
    dataset = dm.train_dataset
    assert dm.shuffle is False  # ensuring same samples over experiments

    # Sample initial data of certain length from dataset sample. Sample num_events samples from one
    # dataset, and create the subsequent event as the one to be added.
    sample = dataset[index % len(dataset)]
    sample.pos = sample.pos[:, :2]
    events_initial = sample_initial_data(sample, num_events, args.radius, edge_attr, args.max_num_neighbors)

    index_new = min(num_events, sample.num_nodes - 1)
    x_new = sample.x[index_new, :].view(1, -1)
    pos_new = sample.pos[index_new, :2].view(1, -1)
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))

    # Initialize model and make it asynchronous (recognition model, so num_outputs = num_classes of input dataset).
    input_shape = torch.tensor([*dm.dims, events_initial.pos.shape[-1]], device=device)
    model = aegnn.models.networks.GraphRes(dm.name, input_shape, dm.num_classes, pooling_size=args.pooling_size)
    model.to(device)
    model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr, **kwargs)

    # Run experiment, i.e. initialize the asynchronous graph and iteratively add events to it.
    _ = model.forward(events_initial.to(device))  # initialization
    _ = model.forward(event_new.to(device))
    del events_initial, event_new
    return model


##################################################################################################
# Logging ########################################################################################
##################################################################################################
def get_log_values(model, attr: str, log_key: str, **log_dict):
    """"Get log values for the given attribute key, both for each layer and in total and for the dense and sparse
    update. Thereby, approximate the logging for the dense with the data for the initial update as
    count(initial events) >>> count(new events)
    """
    assert hasattr(model, attr)
    log_values = []
    for layer, nn in model._modules.items():
        if hasattr(nn, attr):
            logs = getattr(nn, attr)
            log_values.append({"layer": layer, log_key: logs[0], "model": "gnn_dense", **log_dict})
            for log_i in logs[1:]:
                log_values.append({"layer": layer, log_key: log_i, "model": "ours", **log_dict})

    logs = getattr(model, attr)
    log_values.append({"model": "gnn_dense", log_key: logs[0], "layer": "total", **log_dict})
    for log_i in logs[1:]:
        log_values.append({"model": "ours", log_key: log_i, "layer": "total", **log_dict})

    return log_values


##################################################################################################
# Experiments ####################################################################################
##################################################################################################
def run_experiments(dm, args, experiments: List[int], num_trials: int, device: torch.device, **model_kwargs
                    ) -> pd.DataFrame:
    results_df = pd.DataFrame()
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results", "flops.pkl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    runs = list(itertools.product(experiments, list(range(num_trials))))
    for num_events, exp_id in tqdm(runs):
        model = create_and_run_model(dm, num_events, index=exp_id, args=args, device=device, **model_kwargs)

        # Get the logged flops and timings, both layer-wise and in total.
        results_flops = get_log_values(model, attr="asy_flops_log", log_key="flops", num_events=num_events)
        results_runtime = get_log_values(model, attr="asy_runtime_log", log_key="runtime", num_events=num_events)
        results_df = results_df.append(results_flops + results_runtime, ignore_index=True)
        results_df.to_pickle(output_file)

        # Fully reset run to ensure independence between subsequent experiments.
        del model  # fully delete model
        torch.cuda.empty_cache()  # clear memory

    print(f"Results are logged in {output_file}")
    return results_df


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    data_module = aegnn.datasets.NCars(batch_size=1, shuffle=False)
    data_module.setup()
    event_counts = [25000]
    # event_counts = list(np.linspace(1000, 15000, num=10).astype(int))
    run_experiments(data_module, arguments, experiments=event_counts, num_trials=100,
                    device=torch.device(arguments.device), log_flops=True, log_runtime=True)
