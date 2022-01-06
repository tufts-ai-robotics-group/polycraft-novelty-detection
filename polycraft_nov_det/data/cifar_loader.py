from importlib.resources import path

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from polycraft_nov_det.data.base_loader import base_loader


# data shape constant
CIFAR_SHAPE = (3, 32, 32)


def torch_cifar(batch_size=32, include_novel=False, shuffle=True, use_10=True,
                split_seed=42, rot_loader=False):
    """torch DataLoaders for CIFAR10 and CIFAR100

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        use_10 (bool, optional): Use CIFAR10 if True, otherwise CIFAR100. Defaults to True.
        split_seed (int, optional): Seed for splitting normal and novel classes.
                                    Defaults to 42.
        rot_loader (bool, optional): Whether to use RotNet transform. Defaults to False.

    Returns:
        tuple: Returns (norm_targets, novel_targets, dataloaders),
               where dataloaders is a tuple with
               (train_loader, valid_loader, test_loader)
               containing batches of (3, 32, 32) images, with values 0-1.
    """
    dataset_class = CIFAR10 if use_10 else CIFAR100
    num_normal = 5 if use_10 else 50
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
        return base_loader(dataset_class, train_kwargs, test_kwargs, split_seed, num_normal,
                           include_novel, dataloader_kwargs, rot_loader)
