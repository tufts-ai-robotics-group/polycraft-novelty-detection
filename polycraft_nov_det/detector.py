import numpy as np
import torch
from torch.nn.functional import mse_loss

from polycraft_nov_data import data_const as polycraft_const


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

    def value(self, q):
        out = np.array(q * self.slope + self.min)
        if len(out.shape) == 0:
            out = out[np.newaxis]
        return out

    def lt_value(self, data, q):
        thresh_val = self.value(q)
        return data[np.newaxis] < thresh_val[:, np.newaxis]

    def quantile(self, data):
        out = (np.clip(data, self.min, self.max) - self.min) / self.slope
        if len(out.shape) == 0:
            out = out[np.newaxis]
        return out

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
        self.device = torch.device(device)
        self.model = model.to(self.device)
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

    def is_novel_pooled(self, data, quantile=.99, pool_func=np.max):
        """Evaluate novelty based on reconstruction error pooled per batch
        Args:
            data (torch.tensor): Data to use as input to autoencoder and then pool
            quantile (np.ndarray): (N) elements in [0, 1], determines quantile for each output row
            pool_func (func): Function applied to mean reconstruction errors
        Returns:
            np.ndarray: (N) Array of bools, True if data is novel
        """
        return ~self.lin_reg.lt_value(pool_func(self._mean_r_error(data)), quantile)[:, 0]

    def novel_score_pooled(self, data, quantile=.99, pool_func=np.max):
        """Evaluate novelty based on reconstruction error pooled per batch
        Args:
            data (torch.tensor): Data to use as input to autoencoder and then pool
            quantile (np.ndarray): (N) elements in [0, 1], determines quantile for each output row
            pool_func (func): Function applied to mean reconstruction errors
        Returns:
            np.ndarray: (N) Array of scores
        """
        return self.lin_reg.quantile(pool_func(self._mean_r_error(data)))

    def _mean_r_error(self, data):
        # per image mean reconstruction error
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                             (*range(1, data.dim()),))
        return r_error.detach().cpu().numpy()

    def localization(self, data, scale):
        """Evaluate where something (potentially) novel appeared based on where the
           maximum reconstruction error (per patch) appears. Returns 0 if the error
           is highest in the leftmost column/third of the image, 1 if the error is highest in
           the central column/third of the image, 2 if it the error is highest in the
           rightmost column/third.

        Args:
            data (torch.tensor): Data to use as input to autoencoder and then pool

        Returns:
            column (int): 0 --> leftmost column, 1 --> central column,
            2 --> rightmost column
        """
        r_error_per_patch = self._mean_r_error(data)
        # amount of patches per image width
        pw = polycraft_const.IMAGE_SHAPE[1]*scale/(polycraft_const.PATCH_SHAPE[1]//2) - 1
        # amount of patches per image height and width
        two_d_patches_shape = [int(data.shape[0]//pw), int(pw)]
        # reshape flattened patch error array to 2d
        r_error_per_patch = r_error_per_patch.reshape(two_d_patches_shape)
        first_col_idx = int(np.round(two_d_patches_shape[1]/3))
        second_col_idx = int(np.round(two_d_patches_shape[1] - first_col_idx))
        # Average the per patch rec. errors over each corresponding column
        r_error_per_column = np.zeros(3)
        r_error_per_column[0] = np.mean(r_error_per_patch[:, 0:first_col_idx])
        r_error_per_column[1] = np.mean(r_error_per_patch[:, first_col_idx:second_col_idx])
        r_error_per_column[2] = np.mean(r_error_per_patch[:, second_col_idx:two_d_patches_shape[1]])
        # column where maximum rec. error appears
        column = np.argmax(r_error_per_column)
        return column  # 0 --> 1st column, 1 --> 2nd column, 2 --> 3rd column


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


class NoveltyDetector:
    def __init__(self, device="cpu"):
        self.device = device

    @torch.no_grad
    def novelty_score(self, data):
        return 0

    def is_novel(self, data, thresh):
        return self.novelty_score(data) > thresh
