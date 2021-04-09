from importlib.resources import path

import torch
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision import transforms


def torch_mnist(batch_size=32, include_classes=None, shuffle=True):
    """torch DataLoaders for MNIST

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
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
    # select only included classes
    if include_classes is not None:
        train_include = torch.any(torch.stack([train_set.targets == target
                                               for target in include_classes]),
                                  dim=0)
        test_include = torch.any(torch.stack([test_set.targets == target
                                              for target in include_classes]),
                                 dim=0)
        train_set = data.Subset(train_set, torch.nonzero(train_include)[:, 0])
        test_set = data.Subset(test_set, torch.nonzero(test_include)[:, 0])
    # get a validation set
    valid_len = len(train_set) // 10
    train_set, valid_set = data.random_split(train_set, [len(train_set) - valid_len, valid_len],
                                             generator=torch.Generator().manual_seed(42))
    # get DataLoaders for datasets
    return (data.DataLoader(train_set, batch_size, shuffle),
            data.DataLoader(valid_set, batch_size, shuffle),
            data.DataLoader(test_set, batch_size, shuffle))
