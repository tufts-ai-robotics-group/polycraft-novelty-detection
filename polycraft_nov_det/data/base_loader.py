import functools

import torch
from torch.utils import data

import polycraft_nov_data.dataset_transforms as dataset_transforms

import polycraft_nov_det.data.rotnet as rotnet


def reorder_targets(target, target_map):
    return target_map[target]


def base_dataset(dataset_class, train_kwargs, test_kwargs, norm_targets, include_novel):
    """Base dataset generator for novelty related tasks

    Args:
        dataset_class (class): Dataset class to use
        train_kwargs (dict): kwargs for train dataset
        test_kwargs (dict): kwargs for test dataset
        norm_targets (iterable): iterable of ints denoting which targets are in normal set
        include_novel (bool): Whether to include novel data in validation set

    Returns:
        tuple: Returns (norm_targets, novel_targets, datasets),
               where datasets is a tuple with
               (train_set, valid_set, test_set)
    """
    # load datasets
    train_set = dataset_class(**train_kwargs)
    test_set = dataset_class(**test_kwargs)
    # split targets
    targets = [int(target) for target in torch.Tensor(list(train_set.targets)).unique()]
    novel_targets = [target for target in targets if target not in norm_targets]
    class_splits = {key: [1, 0] for key in norm_targets}
    if include_novel:
        class_splits.update({key: [1, 0] for key in novel_targets})
    # reorder targets for cross entropy loss
    target_map = {int(target): i for i, target in enumerate(targets)}
    train_set.target_transform = functools.partial(reorder_targets, target_map=target_map)
    test_set.target_transform = functools.partial(reorder_targets, target_map=target_map)
    # select only included classes and split the train set to get a validation set
    train_set, valid_set = dataset_transforms.filter_split(train_set, class_splits)
    if not include_novel:
        test_set = dataset_transforms.filter_dataset(
            test_set, [test_set.classes[target] for target in norm_targets])
    else:
        test_set = dataset_transforms.filter_dataset(
            test_set, [test_set.classes[target] for target in targets])
    return norm_targets, novel_targets, (train_set, valid_set, test_set)


def base_loader(dataset_class, train_kwargs, test_kwargs, norm_targets, include_novel,
                dataloader_kwargs, rot_loader=None):
    """Base DataLoader generator for novelty related tasks

    Args:
        dataset_class (class): Dataset class to use
        train_kwargs (dict): kwargs for train dataset
        test_kwargs (dict): kwargs for test dataset
        norm_targets (iterable): iterable of ints denoting which targets are in normal set
        include_novel (bool): Whether to include novel data in validation set
        dataloader_kwargs (dict): kwargs for all dataloaders
        rot_loader (bool, optional): Whether to use RotNet transform. Defaults to False.

    Returns:
        tuple: Returns (norm_targets, novel_targets, dataloaders),
               where dataloaders is a tuple with
               (train_loader, valid_loader, test_loader)
    """
    norm_targets, novel_targets, (train_set, valid_set, test_set) = base_dataset(
        dataset_class, train_kwargs, test_kwargs, norm_targets, include_novel)
    # change into RotNet dataset
    if rot_loader is True:
        train_set = rotnet.RotDataset(train_set)
        valid_set = rotnet.RotDataset(valid_set)
        test_set = rotnet.RotDataset(test_set)
        dataloader_kwargs["collate_fn"] = rotnet.collate_fn
    # get DataLoaders for datasets
    dataloaders = (data.DataLoader(train_set, **dataloader_kwargs),
                   data.DataLoader(valid_set, **dataloader_kwargs) if len(valid_set) > 0 else None,
                   data.DataLoader(test_set, **dataloader_kwargs))
    return norm_targets, novel_targets, dataloaders
