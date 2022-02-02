import torch
import torch.utils.data as data
from torchvision.transforms.functional import rotate


class RotDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        target = torch.LongTensor([0, 1, 2, 3])
        img = torch.stack([
            img,
            rotate(img, 90),
            rotate(img, 180),
            rotate(img, 270)
        ], dim=0)
        return img, target


# use collate function to get rid of extra dimension from RotDataset
def collate_fn(batch):
    imgs, targets = data.dataloader.default_collate(batch)
    batch_size, num_rot, channels, height, width = imgs.size()
    imgs = imgs.view([batch_size * num_rot, channels, height, width])
    targets = targets.view([batch_size * num_rot])
    return (imgs, targets)
