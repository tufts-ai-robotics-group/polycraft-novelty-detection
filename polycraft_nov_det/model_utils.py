import torch

from polycraft_nov_det.models.disc_resnet import DiscResNet


def load_model(path, model, device="cpu"):
    # load parameters into a model instance
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_disc_resnet(path, num_labeled_classes, num_unlabeled_classes, device="cpu",
                     reset_head=False, strict=True, to_incremental=False):
    model = DiscResNet(num_labeled_classes, num_unlabeled_classes)
    state_dict = torch.load(path, map_location=device)
    # reset weights for labeled head for self-supervised -> supervised learning
    if reset_head:
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
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
