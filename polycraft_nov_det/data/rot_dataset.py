import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate


class RotDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        target = torch.randint(0, 3, (1,))
        if target == 1:
            data = rotate(data, 90)
        elif target == 2:
            data = rotate(data, 180)
        elif target == 3:
            data = rotate(data, 270)
        return data, target
