from src.dataset import PavlickFormality
from torch.utils.data import DataLoader


class DataModule:
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.dataset = PavlickFormality()
        self.dataloader_params = dict(batch_size=batch_size, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_params)
