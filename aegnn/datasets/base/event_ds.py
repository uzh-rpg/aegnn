from torch_geometric.data import Data
from torch.utils.data.dataset import Dataset, T_co
from typing import Callable, List


class EventDataset(Dataset):

    def __init__(self, files: List[str], load_func: Callable[[str], Data]):
        self.files = files
        self.load_func = load_func

    def __getitem__(self, index: int) -> T_co:
        data_file = self.files[index]
        return self.load_func(data_file)

    def __len__(self) -> int:
        return len(self.files)
