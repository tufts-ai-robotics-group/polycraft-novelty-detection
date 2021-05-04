import os.path

import torch

import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.models.lsa.unmodified.models.LSA_mnist as LSA_mnist
import polycraft_nov_det.models.lsa.unmodified.models.base as base
from polycraft_nov_det.novelty import load_ecdf, reconstruction_ecdf
import polycraft_nov_det.plot as plot


class LSAMNISTNoEst(base.BaseModule):
    """
    LSA model for MNIST one-class classification without estimator.
    """
    def __init__(self,  input_shape, code_length):
        """Class constructor.

        Args:
            input_shape (Tuple[int, int, int]): the shape of MNIST samples.
            code_length (int): the dimensionality of latent vectors.
        """
        super(LSAMNISTNoEst, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        # Build encoder
        self.encoder = LSA_mnist.Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = LSA_mnist.Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

    def forward(self, x):
        """Forward propagation.

        Args:
            x (torch.Tensor): the input batch of images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of torch.Tensors holding reconstructions,
                                               and latent vectors.
        """
        h = x

        # Produce representations
        z = self.encoder(h)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z


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
