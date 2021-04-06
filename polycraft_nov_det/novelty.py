import numpy as np
import torch
from torch.nn.functional import mse_loss


class EmpiricalCDF():
    """Empirical cumulative distribution function
    https://en.wikipedia.org/wiki/Empirical_distribution_function
    """
    def __init__(self, samples):
        """Empirical cumulative distribution function

        Args:
            samples (np.ndarray): Array of (N) samples from distribution to approximate
        """
        self.samples = np.sort(samples)
        self.N = len(self.samples)

    def quantile(self, q):
        """Get q-th quantile of CDF

        Args:
            q (float): In [0, 1], determines quantile
        """
        index = int(np.ceil(self.N * q))
        return self.samples[index]

    def in_quantile(self, data, q):
        """Determine if data is within q-th quantile

        Args:
            data (np.ndarray): array of scalars to evaluate
            q (float): In [0, 1], determines quantile

        Returns:
            np.ndarray: Array of bools same shape as data, True if data is within quantile
        """
        quantile_val = self.quantile(q)
        return data < quantile_val

    def save(self, file):
        np.save(file, self.samples)


def load_ecdf(file):
    return EmpiricalCDF(np.load(file))


class ReconstructionDet():
    """Reconstruction error novelty detector
    """
    def __init__(self, model, ecdf):
        """Reconstruction error novelty detector

        Args:
            model (torch.nn.Module): Autoencoder to measure reconstruction error from
            ecdf (novelty.EmpiricalCDF): ECDF for non-novel reconstruction error
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.ecdf = ecdf

    def is_novel(self, data, q=.99):
        """Evaluate novelty based on reconstruction error

        Args:
            data (torch.tensor): Data to use as input to autoencoder with (N) samples
            q (float, optional): In [0, 1], determines quantile used for evaluation.
                                 Defaults to .99.

        Returns:
            torch.tensor: (N) booleans where True is novel
        """
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                             (*range(1, data.dim()),))
        return ~self.ecdf.in_quantile(r_error.detach().numpy(), q)


def reconstruction_ecdf(model, train_loader):
    """Create an ECDF from autoencoder reconstruction error

    Args:
        model (torch.nn.Module): Autoencoder to measure reconstruction error from
        train_loader (torch.utils.data.Dataloader): Train set for non-novel reconstruction error

    Returns:
        novelty.EmpiricalCDF: ECDF from autoencoder reconstruction error
    """
    device = next(model.parameters()).device
    # get reconstruction error from training data
    r_error = torch.Tensor([])
    for data, _ in train_loader:
        data = data.to(device)
        r_data, _ = model(data)
        # gets mean reconstruction per image in data
        data_r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                                  (*range(1, data.dim()),))
        r_error = torch.cat((r_error, data_r_error))
    # construct EmpiricalCDF from reconstruction error
    return EmpiricalCDF(r_error.detach().numpy())


def reconstruction_ecdf_polycraft(model, train_loader):
    """Create an ECDF from autoencoder reconstruction error for polycraft

    Args:
        model (torch.nn.Module): Autoencoder to measure reconstruction error from
        train_loader (torch.utils.data.Dataloader): Train set for non-novel reconstruction error

    Returns:
        novelty.EmpiricalCDF: ECDF from autoencoder reconstruction error
    """
    device = next(model.parameters()).device
    # get reconstruction error from training data
    r_error = torch.Tensor([])
    for sample in train_loader:
        # TODO revise Polycraft dataloader so preprocessing done by the loader
        # sample contains all patches of one screen image and its novelty description
        patches = sample[0]
        # Dimensions of flat_patches: [1, batch_size, C, H, W]
        flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)
        data = flat_patches[0].float().to(device)
        r_data, _ = model(data)
        # gets mean reconstruction per image in data
        data_r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                                  (*range(1, data.dim()),))
        r_error = torch.cat((r_error, data_r_error))
    # construct EmpiricalCDF from reconstruction error
    return EmpiricalCDF(r_error.detach().numpy())
