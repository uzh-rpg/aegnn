import torch


def yolo_grid(input_shape: torch.Tensor, cell_map_shape: torch.Tensor) -> torch.Tensor:
    """Constructs a 2D grid with the cell center coordinates."""
    cell_shape = input_shape / cell_map_shape
    num_cells = (cell_map_shape * cell_shape).int()
    cell_top_left = torch.meshgrid([torch.arange(0, end=num_cells[0], step=cell_shape[0], device=cell_shape.device),
                                    torch.arange(0, end=num_cells[1], step=cell_shape[1], device=cell_shape.device)])
    return torch.stack(cell_top_left, dim=-1)
