import os.path

import torch

import polycraft_nov_data.data_const as polycraft_const

import polycraft_nov_det.mnist_loader as mnist_loader
from polycraft_nov_det.models.lsa.LSA_mnist_no_est import LSAMNISTNoEst
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst
from polycraft_nov_det.novelty import load_ecdf, reconstruction_ecdf
import polycraft_nov_det.plot as plot


def load_model(path, model, device="cpu"):
    # load parameters into a model instance
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_mnist_model(path, device="cpu", latent_len=64):
    """Load a saved MNIST model

    Args:
        path (str): Path to saved model state_dict
        device (str, optional): Device tag for torch.device. Defaults to "cpu".
        latent_len (int, optional): Length of model's latent vector. Defaults to 64.

    Returns:
        LSAMNISTNoEst: Model with saved state_dict
    """
    model = LSAMNISTNoEst(mnist_loader.MNIST_SHAPE, latent_len)
    return load_model(path, model, device)


def load_polycraft_model(path, device="cpu", latent_len=100):
    """Load a saved Polycraft model

    Args:
        path (str): Path to saved model state_dict
        device (str, optional): Device tag for torch.device. Defaults to "cpu".
        latent_len (int, optional): Length of model's latent vector. Defaults to 100.

    Returns:
        LSACIFAR10NoEst: Model with saved state_dict
    """
    model = LSACIFAR10NoEst(polycraft_const.PATCH_SHAPE, latent_len)
    return load_model(path, model, device)


def load_cached_ecdfs(model_path, model):
    model_dir, model_name = os.path.split(model_path)
    model_name = model_name[:model_name.rfind(".")]
    ecdfs = []
    classes = range(10)
    for i in classes:
        # load cached ECDF if it exists
        ecdf_path = os.path.join(model_dir, "%i_train_%s.npy" % (i, model_name))
        if os.path.isfile(ecdf_path):
            ecdf = load_ecdf(ecdf_path)
        # otherwise generate ECDF and cache it for later
        else:
            train_loader, _, _ = mnist_loader.torch_mnist(include_classes=[i])
            ecdf = reconstruction_ecdf(model, train_loader)
            ecdf.save(ecdf_path)
        ecdfs.append(ecdf)
    return ecdfs


def plot_cached_ecdfs(model_path, model):
    return plot.plot_empirical_cdfs(load_cached_ecdfs(model_path, model), range(10))


def calc_model_embeddings(model, data_loader):
    embeddings = torch.Tensor([])
    targets = torch.Tensor([])
    for data, target in data_loader:
        _, embedding = model(data)
        embeddings = torch.cat((embeddings, embedding))
        targets = torch.cat((targets, target))
    return embeddings, targets
