import pandas as pd


def read_baseline(file_name: str, baseline_name: str):
    flops_df = []
    with open(file_name) as f:
        for line in f.readlines():
            num_nodes = float(line.split(";")[0].replace(",", "."))
            flops = float(line.split(";")[1].replace(",", "."))
            flops_df.append({"num_events": num_nodes, "flops": flops, "model": baseline_name})
    return pd.DataFrame(flops_df)
