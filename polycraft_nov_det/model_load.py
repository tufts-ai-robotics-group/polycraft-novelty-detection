import torch

from polycraft_nov_det.models.dino.hubconf import dino_vitb16


def load_model(path, model, device="cpu"):
    # load parameters into a model instance
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_dino_pretrained(device="cpu"):
    model = dino_vitb16()
    model.to(device)
    return model


def load_dino_block(path, device="cpu"):
    model = dino_vitb16()
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    return model
