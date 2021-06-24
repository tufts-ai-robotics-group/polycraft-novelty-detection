from importlib.resources import path

from torch.utils import data
from torchvision.datasets import MNIST
from torchvision import transforms

import polycraft_nov_data.dataset_transforms as dataset_transforms


# data shape constant
MNIST_SHAPE = (1, 28, 28)


def torch_mnist(batch_size=32, include_novel=False, shuffle=True):
    """torch DataLoaders for MNIST

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.

    Returns:
        (DataLoader, DataLoader, DataLoader): MNIST train, validation, and test sets.
                                              Contains batches of (1, 28, 28) images,
                                              with values 0-1.
    """
    normal_classes = [0, 1, 2, 3, 4]
    novel_classes = [5, 6, 7, 8, 9]
    class_splits = {key: [.9, .1] for key in normal_classes}
    if include_novel:
        class_splits.update({key: [0, 1] for key in novel_classes})
    # get path within module
    with path("polycraft_nov_det", "base_data") as data_path:
        # get dataset with separate data and targets
        train_set = MNIST(root=data_path, train=True, download=True,
                          transform=transforms.ToTensor())
        test_set = MNIST(root=data_path, train=False, download=True,
                         transform=transforms.ToTensor())
    # select only included classes and split the train set to get a validation set
    train_set, valid_set = dataset_transforms.filter_split(train_set, class_splits)
    if not include_novel:
        test_set = dataset_transforms.filter_dataset(test_set, normal_classes)
    else:
        test_set = dataset_transforms.filter_dataset(test_set, normal_classes + novel_classes)
    # get DataLoaders for datasets
    return (data.DataLoader(train_set, batch_size, shuffle),
            data.DataLoader(valid_set, batch_size, shuffle),
            data.DataLoader(test_set, batch_size, shuffle))
