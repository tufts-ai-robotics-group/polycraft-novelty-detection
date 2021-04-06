import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(images, r_images, cmap="rgb"):
    """Plot sample of image reconstructions

    Args:
        images (torch.tensor): Images input to autoencoder
        r_images (torch.tensor): Reconstructed images output from autoencoder
        cmap (str, optional): Color map for images, either "rgb" or "gray". Defaults to "rgb".

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
    # set up axes
    fig, ax = plt.subplots(nrows=2, ncols=num_images)
    ax[0][num_images // 2].set_title("Input")
    ax[1][num_images // 2].set_title("Reconstruction")
    # set up imshow keyword args
    imshow_kwargs = {
        "vmin": 0,
        "vmax": 1,
    }
    if cmap == "gray":
        imshow_kwargs["cmap"] = cmap
    # plot the image and reconstruction side by side
    for i in range(num_images):
        if cmap == "gray":
            ax[0][i].imshow(images[i, 0], **imshow_kwargs)
            ax[1][i].imshow(r_images[i, 0], **imshow_kwargs)
        else:
            # Transpose images in (C, H, W) format to matplotlib's (H, W, C) format
            ax[0][i].imshow(np.transpose(images[i, :], (1, 2, 0)), **imshow_kwargs)
            ax[1][i].imshow(np.transpose(r_images[i, :], (1, 2, 0)), **imshow_kwargs)
    # disable tick marks
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return fig


def plot_empirical_cdfs(ecdfs, labels):
    """Plot CDF of iterable of novelty.EmpiricalCDF

    Args:
        ecdfs (iterable): Iterable of empirical CDFs to plot
        labels (iterable): Iterable of labels for empirical CDFs

    Returns:
        plt.Figure: Figure with plot of CDFs of novelty.EmpiricalCDF
    """
    fig, ax = plt.subplots()
    for ecdf, label in zip(ecdfs, labels):
        # copy of first entry used so plot starts at 0 probability
        plt.plot(np.insert(ecdf.samples, 0, ecdf.samples[0]),
                 np.arange(0, ecdf.N + 1)/ecdf.N,
                 label=label)
        # plot the decision boundary
        plt.scatter(ecdf.quantile(.99), .99)
    plt.legend()
    plt.show()
    return fig
