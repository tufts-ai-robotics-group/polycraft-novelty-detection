import matplotlib.pyplot as plt


def plot_reconstruction(images, r_images):
    # set number of images for plot
    num_images = 5
    if images.shape[0] < num_images:
        num_images = images.shape[0]
    # remove grad from tensors for numpy conversion
    images = images.detach()
    r_images = r_images.detach()
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
