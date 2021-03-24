import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(images, r_images):
    """Plot sample of image reconstructions

    Args:
        images (torch.tensor): Images input to autoencoder
        r_images (torch.tensor): Reconstructed images output from autoencoder

    Returns:
        plt.Figure: Figure with plot of sample of image reconstructions
    """
    # set number of images for plot
    num_images = 5
    if images.shape[0] < num_images:
        num_images = images.shape[0]
    # remove grad from tensors for numpy conversion
    images = images.detach().cpu()
    r_images = r_images.detach().cpu()
    # plot the image and reconstruction side by side
    fig, ax = plt.subplots(nrows=2, ncols=num_images)
    ax[0][num_images // 2].set_title("Input")
    ax[1][num_images // 2].set_title("Reconstruction")
    imshow_kwargs = {
        "cmap": "gray",
        "vmin": 0,
        "vmax": 1,
    }
    for i in range(num_images):
        ax[0][i].imshow(images[i, 0], **imshow_kwargs)
        ax[1][i].imshow(r_images[i, 0], **imshow_kwargs)
    # disable tick marks
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return fig


def plot_reconstruction_rgb(images, r_images):
    # set number of images for plot
    num_images = 5
    if images.shape[0] < num_images:
        num_images = images.shape[0]
    # remove grad from tensors for numpy conversion
    images = images.detach().cpu()
    r_images = r_images.detach().cpu()
    # plot the image and reconstruction side by side
    fig, ax = plt.subplots(nrows=2, ncols=num_images)
    ax[0][num_images // 2].set_title("Input")
    ax[1][num_images // 2].set_title("Reconstruction")
    imshow_kwargs = {
        "vmin": 0,
        "vmax": 1,
    }

    for i in range(num_images):
        #Transpose images in (C, H, W) format to matplotlib's (H, W, C) format
        ax[0][i].imshow(np.transpose(images[i, :], (1, 2, 0)))
        ax[1][i].imshow(np.transpose(r_images[i, :], (1, 2, 0)))
    # disable tick marks
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return fig


def plot_empirical_cdf(ecdf):
    """Plot CDF of novelty.EmpiricalCDF

    Args:
        ecdf (novelty.EmpiricalCDF): Empirical CDF to plot

    Returns:
        plt.Figure: Figure with plot of CDF of novelty.EmpiricalCDF
    """
    fig, ax = plt.subplots()
    # copy of first entry used so plot starts at 0 probability
    plt.plot(np.insert(ecdf.samples, 0, ecdf.samples[0]),
             np.arange(0, ecdf.N + 1)/ecdf.N)
    plt.show()
    return fig
