import torch
from torch.utils import data

import polycraft_nov_data.dataset_transforms as dataset_transforms

from polycraft_nov_det.data.rot_dataset import RotDataset, RotConsistentDataset


def base_dataset(dataset_class, train_kwargs, test_kwargs, split_seed, num_normal, include_novel):
    """Base dataset generator for novelty related tasks

    Args:
        dataset_class (class): Dataset class to use
        train_kwargs (dict): kwargs for train dataset
        test_kwargs (dict): kwargs for test dataset
        split_seed (int): Seed for splitting normal and novel classes
        num_normal (int): Number of classes to label normal
        include_novel (bool): Whether to include novel data in validation set

    Returns:
        tuple: Returns (norm_targets, novel_targets, datasets),
               where datasets is a tuple with
               (train_set, valid_set, test_set)
    """
    # initialize seed
    split_gen = torch.manual_seed(split_seed)
    # load datasets
    train_set = dataset_class(**train_kwargs)
    test_set = dataset_class(**test_kwargs)
    # split targets
    targets = torch.Tensor(list(train_set.targets)).unique()
    targets = targets[torch.randperm(len(targets), generator=split_gen)]
    norm_targets = targets[:num_normal]
    novel_targets = targets[num_normal:]
    class_splits = {key: [.9, .1] for key in norm_targets}
    if include_novel:
        class_splits.update({key: [.9, .1] for key in novel_targets})
    # reorder targets for cross entropy loss
    target_map = {int(target): i for i, target in enumerate(targets)}

    def reorder_targets(target):
        return target_map[target]
    train_set.target_transform = reorder_targets
    test_set.target_transform = reorder_targets
    # select only included classes and split the train set to get a validation set
    train_set, valid_set = dataset_transforms.filter_split(train_set, class_splits)
    if not include_novel:
        test_set = dataset_transforms.filter_dataset(test_set, norm_targets)
    else:
        test_set = dataset_transforms.filter_dataset(test_set, targets)
    return norm_targets, novel_targets, (train_set, valid_set, test_set)


def base_loader(dataset_class, train_kwargs, test_kwargs, split_seed, num_normal, include_novel,
                dataloader_kwargs, rot_loader=None):
    """Base DataLoader generator for novelty related tasks

    Args:
        dataset_class (class): Dataset class to use
        train_kwargs (dict): kwargs for train dataset
        test_kwargs (dict): kwargs for test dataset
        split_seed (int): Seed for splitting normal and novel classes
        num_normal (int): Number of classes to label normal
        include_novel (bool): Whether to include novel data in validation set
        dataloader_kwargs (dict): kwargs for all dataloaders
        rot_loader (str, optional): Whether to use RotNet transform. Defaults to None.
                                    "rotnet" applies random rotation and with rotation labels.
                                    "consistent" provides (image, rotated image, original label)
        rot_loader (bool, optional): Whether to use RotNet transform. Defaults to False.

    Returns:
        tuple: Returns (norm_targets, novel_targets, dataloaders),
               where dataloaders is a tuple with
               (train_loader, valid_loader, test_loader)
    """
    norm_targets, novel_targets, (train_set, valid_set, test_set) = base_dataset(
        dataset_class, train_kwargs, test_kwargs, split_seed, num_normal, include_novel)
    if rot_loader is None:
        pass
    elif rot_loader == "rotnet":
        train_set = RotDataset(train_set)
        valid_set = RotDataset(valid_set)
        test_set = RotDataset(test_set)
    elif rot_loader == "consistent":
        train_set = RotConsistentDataset(train_set)
        valid_set = RotConsistentDataset(valid_set)
        test_set = RotConsistentDataset(test_set)
    else:
        raise Exception("Unexpected rot_loader value: " + str(rot_loader))
    # get DataLoaders for datasets
    dataloaders = (data.DataLoader(train_set, **dataloader_kwargs),
                   data.DataLoader(valid_set, **dataloader_kwargs),
                   data.DataLoader(test_set, **dataloader_kwargs))
    return norm_targets, novel_targets, dataloaders
