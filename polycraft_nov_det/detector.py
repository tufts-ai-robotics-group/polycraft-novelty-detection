import numpy as np
import torch
from torch.nn.functional import mse_loss
from polycraft_nov_data.data_const import PATCH_SHAPE, IMAGE_SHAPE
from torchvision.transforms import functional


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

    def _mean_r_error(self, data):
        # per image mean reconstruction error
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                             (*range(1, data.dim()),))
        return r_error.detach().cpu().numpy()
    
    
class ReconstructionDetMultiScale():
    """Multiscale reconstruction error novelty detector
    """
    def __init__(self, classifier, models_allscales, device="cpu"):
        """Multiscale reconstruction error novelty detector
        Previously trained binary classifier is applied on the rec.error -
        arrays of each scale.

        Args:
            classifier (torch.nn.Module): Trained binary classifier
            models_allscales (torch.nn.Module): List of autoencoder to measure reconstruction error from
            device (str, optional): Device tag for torch.device. Defaults to "cpu"
            autoencoder model trained at scale 1 and 3x15x16 patches is used
        """
        self.device = torch.device(device)
        self.classifier = classifier
        self.models_allscales = models_allscales
        
    def is_novel(self, data_multiscale):
        """Evaluate novelty based on reconstruction error per image at each scale

        Args:
            data_multiscale (list of torch.tensor): Data to use as input to autoencoder,
            the image is represented in patches at each scale (0.5, 0.75, 1)
        Returns:
            novelty_score (float): float value in range [0, 1]. The higher, the
            "more novel"
        """
        # shapes of "patch array" for all scales + one for the 16x16 model
        ipt_shapes = [[6, 7],  # Scale 0.5, patch size 3x32x32
                      [9, 11],  # Scale 0.75, patch size 3x32x32
                      [13, 15],  # Scale 1, patch size 3x32x32
                      [28, 31]]  # Scale 1, patch size 3x16x316
        
        with torch.no_grad():
            
            loss_arrays = ()

            for n, model in enumerate(self.models_allscales):
                
                data = data_multiscale[n][0][0].float().to(self.device)
                data = torch.flatten(data, start_dim=0, end_dim=1)
                _, ih, iw = IMAGE_SHAPE
                _, ph, pw = PATCH_SHAPE
                ipt_shape = ipt_shapes[n]
                
                r_data, z = model(data)
                # #patches x 3 x 32 x 32
                loss2d = mse_loss(data, r_data, reduction="none")
                loss2d = torch.mean(loss2d, (2, 3))  # avgd. per patch
                # Reshape loss values from flattened to "squared shape"
                loss2d_r = loss2d[:, 0].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                loss2d_g = loss2d[:, 1].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                loss2d_b = loss2d[:, 2].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                # Concatenate to a "3 channel" loss array
                loss2d_rgb = torch.cat((loss2d_r, loss2d_g), 1)
                loss2d_rgb = torch.cat((loss2d_rgb, loss2d_b), 1)
                # Interpolate smaller scales such that they match scale 1
                loss2d = functional.resize(loss2d_rgb, (13, 15))
                loss_arrays = loss_arrays + (loss2d,)
        
            # Classifier was trained over array of RGB losses
            novelty_score = self.classifier(loss_arrays)
        
        return novelty_score


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
