import torch

from polycraft_nov_det.models.autonovel_resnet import AutoNovelResNet
from polycraft_nov_det.models.dino.hubconf import dino_vitb16


def load_model(path, model, device="cpu"):
    # load parameters into a model instance
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_autonovel_resnet(path, num_labeled_classes, num_unlabeled_classes, device="cpu",
                          reset_head=False, strict=True, to_incremental=False):
    model = AutoNovelResNet(num_labeled_classes, num_unlabeled_classes)
    state_dict = torch.load(path, map_location=device)
    # reset weights for labeled head for self-supervised -> supervised learning
    if reset_head:
        del state_dict["head1.weight"]
        del state_dict["head1.bias"]
    # remove empty tensors to stop errors when strict=False
    if strict is False:
        keys = [key for key in state_dict]  # copy keys so dict can be modified in place
        for key in keys:
            val = state_dict[key]
            if len(val.shape) > 0 and len(val) == 0:
                del state_dict[key]
    # load parameters
    model.load_state_dict(state_dict, strict=strict)
    # to transfer to incremental learning freeze most parameters and intialize labeled head
    if to_incremental:
        model.freeze_layers()
        model.init_incremental()
    return model


def load_autonovel_pretrained(path, num_labeled_classes, num_unlabeled_classes, device="cpu"):
    model = AutoNovelResNet(num_labeled_classes, num_unlabeled_classes)
    state_dict = torch.load(path, map_location=device)
    # if loading a self-supervised model
    if num_unlabeled_classes == 0:
        # swap name of first head
        state_dict["head1.weight"] = state_dict.pop("linear.weight")
        state_dict["head1.bias"] = state_dict.pop("linear.bias")
        # add empty params for second head
        state_dict["head2.weight"] = model.state_dict()["head2.weight"]
        state_dict["head2.bias"] = model.state_dict()["head2.bias"]
    # load parameters
    model.load_state_dict(state_dict)
    return model


def load_dino_pretrained(device="cpu"):
    model = dino_vitb16()
    model.to(device)
    return model
