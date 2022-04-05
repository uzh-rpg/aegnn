import torch


def compute_flops_voxel_grid(pos: torch.Tensor) -> int:
    return pos.numel()
