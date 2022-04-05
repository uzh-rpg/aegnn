import numpy as np
import torch
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

import matplotlib.pyplot as plt

# TODO: add and replace with visualization notebooks

num_nodes = 100
space = 20
spt_distance = 3.0
num_steps = 4
growths = [1, 3]

# Old events generation.
x = torch.tensor(np.random.choice([-1, 1], num_nodes)).view(-1, 1).float()
pos = torch.rand(num_nodes, 2) * space
y = torch.tensor(np.random.choice([0, 1], num_nodes)).long()
data = Data(x=x, pos=pos, y=y)

# New event generation.
event = Data(x=torch.tensor(1).view(-1, 1), pos=torch.rand(1, 2) * space)
pos = torch.cat([data.pos, event.pos])
edges = radius_graph(pos, r=spt_distance, max_num_neighbors=pos.size()[0])

fig, ax = plt.subplots(len(growths), num_steps, figsize=(num_steps * 5, len(growths) * 5))
node_idx = pos.size()[0] - 1
for i, g in enumerate(growths):
    for j in range(num_steps):
        subset, _, _, _ = torch_geometric.utils.k_hop_subgraph(node_idx, num_hops=g * j, edge_index=edges)

        for edge in edges.T:
            pos_edge = pos[[edge[0], edge[1]], :]
            ax[i, j].plot(pos_edge[:, 0], pos_edge[:, 1], "k-", linewidth=0.1)
        ax[i, j].plot(pos[:, 0], pos[:, 1], "bo")
        ax[i, j].plot(pos[subset, 0], pos[subset, 1], "ro")

        ax[i, j].set_title(f"Layer = {j} with K = {g * j}")
fig.savefig("k_hop_graph.png")
