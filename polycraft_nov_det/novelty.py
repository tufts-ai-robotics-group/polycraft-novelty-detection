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
    def __init__(self, model, ecdf, device="cpu"):
        """Reconstruction error novelty detector

        Args:
            model (torch.nn.Module): Autoencoder to measure reconstruction error from
            ecdf (novelty.EmpiricalCDF): ECDF for non-novel reconstruction error
            device (str, optional): Device tag for torch.device. Defaults to "cpu".
        """
        self.model = model
        self.device = torch.device(device)
        self.ecdf = ecdf

    def is_novel(self, data, quantile=.99):
        """Evaluate novelty based on reconstruction error per image

        Args:
            data (torch.tensor): Data to use as input to autoencoder with (B) samples
            quantile (float, optional): In [0, 1], determines quantile used for evaluation.
                                        Defaults to .99.

        Returns:
            np.ndarray: (B) booleans where True is novel
        """
        return ~self.ecdf.in_quantile(self._mean_r_error(data), quantile)

    def is_novel_pooled(self, data, pool_func=np.max, quantile=.99):
        """Evaluate novelty based on reconstruction error pooled per batch

        Args:
            data (torch.tensor): Data to use as input to autoencoder with (B) samples
            pool_func (func): Function applied to (B) mean reconstruction errors.
            quantile (float, optional): In [0, 1], determines quantile used for evaluation.
                                        Defaults to .99.

        Returns:
            np.ndarray: boolean where True is novel
        """
        return ~self.ecdf.in_quantile(pool_func(self._mean_r_error(data)), quantile)

    def _mean_r_error(self, data):
        # per image mean reconstruction error
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                             (*range(1, data.dim()),))
        return r_error.detach().numpy()


def reconstruction_ecdf(model, train_loader, device="cpu"):
    """Create an ECDF from autoencoder reconstruction error

    Args:
        model (torch.nn.Module): Autoencoder to measure reconstruction error from
        train_loader (torch.utils.data.Dataloader): Train set for non-novel reconstruction error
        device (str, optional): Device tag for torch.device. Defaults to "cpu".

    Returns:
        novelty.EmpiricalCDF: ECDF from autoencoder reconstruction error
    """
    device = torch.device(device)
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
