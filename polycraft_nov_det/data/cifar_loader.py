from importlib.resources import path

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from polycraft_nov_det.data.base_loader import base_loader
from polycraft_nov_det.data.loader_trans import GaussianBlur, TransformTwice


# data shape constant
CIFAR_SHAPE = (3, 32, 32)
IMAGENET_SHAPE = (3, 224, 224)


def torch_cifar(norm_targets, batch_size=32, include_novel=False, shuffle=True, use_10=True,
                rot_loader=None):
    """torch DataLoaders for CIFAR10 and CIFAR100

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        include_novel (bool, optional): Whether to include novelties in non-train sets.
                                        Defaults to False.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        use_10 (bool, optional): Use CIFAR10 if True, otherwise CIFAR100. Defaults to True.
        split_seed (int, optional): Seed for splitting normal and novel classes.
                                    Defaults to 42.
        rot_loader (str, optional): Whether to use RotNet transform. Defaults to None.
                                    "rotnet" applies random rotation and with rotation labels.
                                    "consistent" provides
                                    ((image, transformed image), original label)

    Returns:
        tuple: Returns (norm_targets, novel_targets, dataloaders),
               where dataloaders is a tuple with
               (train_loader, valid_loader, test_loader)
               containing batches of (3, 32, 32) images, with values 0-1.
    """
    dataset_class = CIFAR10 if use_10 else CIFAR100
    with path("polycraft_nov_det", "base_data") as data_path:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        interp = transforms.InterpolationMode.BICUBIC
        # resizing to ImageNet shape (3, 224, 224)
        resize = transforms.Resize(IMAGENET_SHAPE[1], interpolation=interp)
        if rot_loader == "consistent":
            # based on DINO global transforms
            # https://github.com/facebookresearch/dino/blob/main/main_dino.py
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            # guaranteed blur transform
            blur_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGENET_SHAPE[1], scale=(.4, 1),
                                             interpolation=interp),
                flip_and_color_jitter,
                GaussianBlur(1.0),
            ])
            # chance of blur and/or solarize transform
            blur_solarize_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGENET_SHAPE[1], scale=(.4, 1),
                                             interpolation=interp),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                transforms.RandomSolarize(128, p=.2),
            ])
            # randomly apply one of two above
            train_transform = TransformTwice(transforms.Compose([
                resize,
                transforms.RandomChoice([blur_transform, blur_solarize_transform]),
                test_transform,
            ]))
        else:
            train_transform = transforms.Compose([
                resize,
                transforms.RandomResizedCrop(IMAGENET_SHAPE[1], scale=(.4, 1),
                                             interpolation=interp),
                transforms.RandomHorizontalFlip(),
                test_transform,
            ])
        train_kwargs = {
            "root": data_path,
            "train": True,
            "download": True,
            "transform": train_transform
        }
        test_kwargs = {
            "root": data_path,
            "train": False,
            "download": True,
            "transform": test_transform
        }
        num_workers = min(4, batch_size)
        dataloader_kwargs = {
            "num_workers": num_workers,
            "prefetch_factor": batch_size//num_workers,
            "persistent_workers": True,
            "batch_size": batch_size,
            "shuffle": shuffle,
        }
        return base_loader(dataset_class, train_kwargs, test_kwargs, norm_targets, include_novel,
                           dataloader_kwargs, rot_loader == "rotnet")
