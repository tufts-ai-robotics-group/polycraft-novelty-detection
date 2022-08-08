import torch

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst
from polycraft_nov_det.models.vgg import VGGPretrained


def load_model(path, model, device="cpu"):
    # load parameters into a model instance
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_autoencoder_model(path, input_shape, device="cpu", latent_len=100):
    """Load a saved Polycraft model
    Args:
        path (str): Path to saved model state_dict
        device (str, optional): Device tag for torch.device. Defaults to "cpu".
        latent_len (int, optional): Length of model's latent vector. Defaults to 100.
    Returns:
        LSACIFAR10NoEst: Model with saved state_dict
    """
    model = LSACIFAR10NoEst(input_shape, latent_len)
    return load_model(path, model, device)


def load_vgg_model(path, device="cpu", num_classes=5):
    model = VGGPretrained(num_classes)
    return load_model(path, model, device)


def calc_model_embeddings(model, data_loader):
    embeddings = torch.Tensor([])
    targets = torch.Tensor([])
    for data, target in data_loader:
        _, embedding = model(data)
        embeddings = torch.cat((embeddings, embedding))
        targets = torch.cat((targets, target))
    return embeddings, targets
