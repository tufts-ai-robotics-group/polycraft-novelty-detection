import numpy as np
import torch
from torch.nn.functional import mse_loss


class LinearRegularization():
    """Linear regularization for thresholding reconstruction error
    """
    def __init__(self, extrema):
        """Linear regularization for thresholding reconstruction error

        Args:
            extrema (np.ndarray): Array [min, max] of reconstruction error set to regularize
        """
        self.extrema = extrema
        self.min = extrema[0]
        self.max = extrema[1]
        self.slope = self.max - self.min

    def value(self, x):
        return x * self.slope + self.min

    def lt_value(self, data, x):
        y = self.value(x)
        return data[np.newaxis] < y[:, np.newaxis]

    def save(self, file):
        np.save(file, self.extrema)


def load_lin_reg(file):
    return LinearRegularization(np.load(file))


class ReconstructionDet():
    """Reconstruction error novelty detector
    """
    def __init__(self, model, lin_reg, device="cpu"):
        """Reconstruction error novelty detector

        Args:
            model (torch.nn.Module): Autoencoder to measure reconstruction error from
            lin_reg (novelty.LinearRegularization): Regularization for non-novel error
            device (str, optional): Device tag for torch.device. Defaults to "cpu"
        """
        self.model = model
        self.device = torch.device(device)
        self.lin_reg = lin_reg

    def is_novel(self, data, quantile=.99):
        """Evaluate novelty based on reconstruction error per image

        Args:
            data (torch.tensor): Data to use as input to autoencoder with (B) samples
            quantile (np.ndarray): (N) elements in [0, 1], determines quantile for each output row

        Returns:
            np.ndarray: (N, B) Array of bools, True if data is novel
        """
        return ~self.lin_reg.lt_value(self._mean_r_error(data), quantile)

    def is_novel_pooled(self, data, pool_func=np.max, quantile=.99):
        """Evaluate novelty based on reconstruction error pooled per batch

        Args:
            data (torch.tensor): Data to use as input to autoencoder and then pool
            quantile (np.ndarray): (N) elements in [0, 1], determines quantile for each output row
            pool_func (func): Function applied to mean reconstruction errors

        Returns:
            np.ndarray: (N) Array of bools, True if data is novel
        """
        return ~self.lin_reg.lt_value(pool_func(self._mean_r_error(data)), quantile)[:, 0]

    def _mean_r_error(self, data):
        # per image mean reconstruction error
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                             (*range(1, data.dim()),))
        return r_error.detach().numpy()


def reconstruction_lin_reg(model, train_loader, device="cpu"):
    """Create an linear regularization from autoencoder reconstruction error

    Args:
        model (torch.nn.Module): Autoencoder to measure reconstruction error from
        train_loader (torch.utils.data.Dataloader): Train set for non-novel reconstruction error
        device (str, optional): Device tag for torch.device. Defaults to "cpu".

    Returns:
        novelty.EmpiricalCDF: Linear regularization from autoencoder reconstruction error
    """
    device = torch.device(device)
    # get reconstruction error from training data
    train_min = None
    train_max = None
    for data, _ in train_loader:
        data = data.to(device)
        r_data, _ = model(data)
        # gets mean reconstruction per image in data
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"), (*range(1, data.dim()),))
        # store the min and max reconstruction errors
        if train_min is None or train_min > torch.min(r_error):
            train_min = torch.min(r_error).item()
        if train_max is None or train_max < torch.max(r_error):
            train_max = torch.max(r_error).item()
    # construct EmpiricalCDF from reconstruction error
    return LinearRegularization(np.array([train_min, train_max]))
