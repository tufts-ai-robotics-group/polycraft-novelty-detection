import os.path

import torch

import polycraft_nov_det.mnist_loader as mnist_loader
from polycraft_nov_det.models.lsa.LSA_mnist_no_est import LSAMNISTNoEst
from polycraft_nov_det.novelty import load_ecdf, reconstruction_ecdf
import polycraft_nov_det.plot as plot


def load_model(path):
    """Load a saved model

    Args:
        path (str): Path to saved model state_dict

    Returns:
        LSAMNISTNoEst: Model with saved state_dict
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = LSAMNISTNoEst(mnist_loader.MNIST_SHAPE, 64)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_cached_ecdfs(model_path):
    model = load_model(model_path)
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


def plot_cached_ecdfs(model_path):
    return plot.plot_empirical_cdfs(load_cached_ecdfs(model_path), range(10))


def plot_embedding(model_path):
    model = load_model(model_path)
    embeddings = torch.Tensor([])
    targets = torch.Tensor([])
    # get embedding and targets from validation set
    _, valid_loader, _ = mnist_loader.torch_mnist()
    for data, target in valid_loader:
        _, embedding = model(data)
        embeddings = torch.cat((embeddings, embedding))
        targets = torch.cat((targets, target))
    # plot the embedding
    plot.plot_embedding(embeddings, targets)
