from importlib.resources import path

import torch
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision import transforms


def torch_mnist(batch_size=1, include_classes=None, shuffle=True):
    """torch DataLoaders for MNIST

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 1.
        include_classes (list, optional): List of classes to include.
                                          Defaults to None, including all classes.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.

    Returns:
        (DataLoader, DataLoader, DataLoader): MNIST train, validation, and test sets.
                                              Contains batches of (1, 28, 28) images,
                                              with values 0-1.
    """
    # get path within module
    with path("polycraft_nov_det", "base_data") as data_path:
        # get dataset with separate data and targets
        train_set = MNIST(root=data_path, train=True, download=True,
                          transform=transforms.ToTensor())
        test_set = MNIST(root=data_path, train=False, download=True,
                         transform=transforms.ToTensor())
    # sort the training set by class
    class_count = torch.bincount(train_set.targets)
    sort_ind = torch.argsort(train_set.targets)
    train_set = data.Subset(train_set, sort_ind)
    # get a validation set with an even class distribution
    valid_count = class_count // 10
    train_ind = torch.tensor([], dtype=torch.int32)
    valid_ind = torch.tensor([], dtype=torch.int32)
    for i in range(len(class_count)):
        class_start = torch.sum(class_count[:i])
        class_end = torch.sum(class_count[:i + 1])
        train_ind = torch.cat((train_ind, torch.arange(class_start, class_end - valid_count[i])))
        valid_ind = torch.cat((valid_ind, torch.arange(class_end - valid_count[i], class_end)))
    valid_set = data.Subset(train_set, valid_ind)
    train_set = data.Subset(train_set, train_ind)
    # get DataLoaders for datasets
    return (data.DataLoader(train_set, batch_size, shuffle),
            data.DataLoader(valid_set, batch_size, shuffle),
            data.DataLoader(test_set, batch_size, shuffle))
