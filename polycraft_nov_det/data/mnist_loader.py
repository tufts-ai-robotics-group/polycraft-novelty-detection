from importlib.resources import path

from torchvision.datasets import MNIST
from torchvision import transforms

from polycraft_nov_det.data.base_loader import base_loader


# data shape constant
MNIST_SHAPE = (1, 28, 28)


def torch_mnist(batch_size=32, include_novel=False, shuffle=True,
                split_seed=42):
    """torch DataLoaders for MNIST

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        split_seed (int, optional): Seed for splitting normal and novel classes.
                                    Defaults to 42.

    Returns:
        (DataLoader, DataLoader, DataLoader): MNIST train, validation, and test sets.
                                              Contains batches of (1, 28, 28) images,
                                              with values 0-1.
    """
    dataset_class = MNIST
    num_normal = 5
    with path("polycraft_nov_det", "base_data") as data_path:
        train_kwargs = {
            "root": data_path,
            "train": True,
            "download": True,
            "transform": transforms.ToTensor()
        }
        test_kwargs = {
            "root": data_path,
            "train": False,
            "download": True,
            "transform": transforms.ToTensor()
        }
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
        }
        return base_loader(dataset_class, train_kwargs, test_kwargs, dataloader_kwargs,
                           split_seed, num_normal, include_novel)
