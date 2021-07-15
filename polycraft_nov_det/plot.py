import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    # determine the color map to use
    if images.shape[1] == 1:
        cmap = "gray"
    else:
        cmap = "rgb"
    # remove grad from tensors for numpy conversion
    images = np.clip(images.detach().cpu(), 0, 1)
    r_images = np.clip(r_images.detach().cpu(), 0, 1)
    # set up axes
    fig, ax = plt.subplots(nrows=2, ncols=num_images)
    ax[0][num_images // 2].set_title("Input")
    ax[1][num_images // 2].set_title("Reconstruction")
    # set up imshow keyword args
    imshow_kwargs = {
        "vmin": 0,
        "vmax": 1,
    }
    # plot the image and reconstruction side by side
    for i in range(num_images):
        if cmap == "gray":
            ax[0][i].imshow(images[i, 0], cmap=cmap, **imshow_kwargs)
            ax[1][i].imshow(r_images[i, 0], cmap=cmap, **imshow_kwargs)
        else:
            # Transpose images in (C, H, W) format to matplotlib's (H, W, C) format
            ax[0][i].imshow(np.transpose(images[i, :], (1, 2, 0)), **imshow_kwargs)
            ax[1][i].imshow(np.transpose(r_images[i, :], (1, 2, 0)), **imshow_kwargs)
    # disable tick marks
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return fig


def plot_empirical_cdf(ecdf, opt_thresh):
    """Plot CDF of novelty.EmpiricalCDF

    Args:
        ecdf (novelty.EmpiricalCDF): Empirical CDF to plot
        opt_thresh (float): Optimal threshold found by eval_calc.optimal_index

    Returns:
        plt.Figure: Figure with plot of CDFs of novelty.EmpiricalCDF
    """
    fig, ax = plt.subplots()
    # copy of first entry used so plot starts at 0 probability
    plt.plot(np.insert(ecdf.samples, 0, ecdf.samples[0]),
             np.arange(0, ecdf.N + 1)/ecdf.N)
    # plot the decision boundary
    plt.scatter(ecdf.quantile(np.array([opt_thresh])), opt_thresh)
    plt.title("Empirical CDF")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Training Set Percentile")
    return fig


def plot_per_patch_nov_det(detector, quantile, patch_array_shape, all_patches_batch):
    """Plot novelty detection for each patch of an image

    Args:
        detector (novelty.ReconstructionDet): Detector to make novelty decisions
        quantile (float): Quantile to use for detector
        patch_array_shape (tuple): Shape of the output of ToPatches, (PH, PW, C, H, W)
        all_patches_batch (torch.Tensor): Batch from a DataLoader with all_patches=True,
                                          shape (PH * PW, C, H, W)

    Returns:
        plt.Figure: Figure with plot of novelty detection for each patch of an image
    """
    patch_h, patch_w, _, _, _ = patch_array_shape
    fig = plt.figure()
    # get novelty detections
    is_novel = detector.is_novel(all_patches_batch, quantile)
    for i in range(all_patches_batch.shape[0]):
        # create patch subplot with detection as title
        ax = fig.add_subplot(patch_h, patch_w, i + 1)
        ax.set_title(is_novel[i])
        # plot the patch
        img = all_patches_batch[i]
        plt.imshow(np.transpose(img.detach().numpy(), (1, 2, 0)))
        plt.subplots_adjust(hspace=1)
        plt.tick_params(top=False, bottom=False, left=False,
                        right=False, labelleft=False, labelbottom=False)
    fig.suptitle("Patches with Novelty Labels")
    return fig


def plot_embedding(embeddings, targets):
    """Plot PCA projection of autoencoder embeddings

    Args:
        embeddings (torch.tensor): (N, D) autoencoder embeddings
        targets (torch.tensor): (N) labels for data that was embedded

    Returns:
        plt.Figure: Figure with PCA projection of autoencoder embeddings
    """
    # remove grad from tensors for numpy conversion
    embeddings = embeddings.detach().cpu()
    targets = targets.detach().cpu()
    # get projection of embeddings
    pca = PCA(2, random_state=0)
    embeddings_proj = pca.fit_transform(embeddings)
    # labeled scatter plot
    fig, ax = plt.subplots()
    for target in np.unique(targets):
        is_target = targets == target
        plt.scatter(embeddings_proj[is_target, 0], embeddings_proj[is_target, 1], label=target)
    plt.legend()
    return fig
