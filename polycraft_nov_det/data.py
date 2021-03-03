from importlib.resources import path

import torch
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision import transforms


def torch_mnist(batch_size=1, shuffle=True):
    """torch DataLoaders for MNIST

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 1.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.

    Returns:
        (DataLoader, DataLoader, DataLoader): MNIST train, validation, and test sets.
                                              Contains batches of (1, 28, 28) images,
                                              with values 0-255.
    """
    # get path within module
    with path("polycraft_nov_det", "base_data") as data_path:
        # get dataset with separate data and targets
        train_set = MNIST(root=data_path, train=True, download=True,
                          target_transform=transforms.ToTensor())
        test_set = MNIST(root=data_path, train=False, download=True,
                         target_transform=transforms.ToTensor())
    # sort the training set by class
    sort_ind = torch.argsort(train_set.targets)
    train_data = train_set.data[sort_ind]
    train_targets = train_set.targets[sort_ind]
    # adds extra channel dimension to data for compatibility with networks expecting RGB
    train_data = train_set.data[:, None].float()
    test_data = test_set.data[:, None].float()
    # get a validation set with an even class distribution
    class_count = torch.bincount(train_targets)
    valid_count = class_count // 10
    valid_mask = torch.zeros(len(train_targets), dtype=torch.bool)
    class_end = 0
    for i in range(len(class_count)):
        class_end += class_count[i]
        valid_mask[torch.arange(class_end - valid_count[i], class_end)] = True
    valid_data = train_data[valid_mask]
    valid_targets = train_targets[valid_mask]
    train_data = train_data[~valid_mask]
    train_targets = train_targets[~valid_mask]
    # get tensor dataset with data and targets
    train_tensor = data.TensorDataset(train_data, train_targets)
    valid_tensor = data.TensorDataset(valid_data, valid_targets)
    test_tensor = data.TensorDataset(test_data, test_set.targets)
    # get DataLoaders for datasets
    return (data.DataLoader(train_tensor, batch_size, shuffle),
            data.DataLoader(valid_tensor, batch_size, shuffle),
            data.DataLoader(test_tensor, batch_size, shuffle))
