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
        target = torch.randint(0, 3, (1,))[0]
        if target == 1:
            data = rotate(data, 90)
        elif target == 2:
            data = rotate(data, 180)
        elif target == 3:
            data = rotate(data, 270)
        return data, target


class RotConsistentDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        rot = torch.randint(1, 3, (1,))[0]
        if rot == 1:
            rot_data = rotate(data, 90)
        elif rot == 2:
            rot_data = rotate(data, 180)
        elif rot == 3:
            rot_data = rotate(data, 270)
        return data, rot_data, target
