import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA


def plot_con_matrix(con_matrix):
    disp = metrics.ConfusionMatrixDisplay(con_matrix, display_labels=["Novel", "Normal"])
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


def plot_gcd_con_matrix(con_matrix):
    labels = ["Normal", "Fence", "Anvil", "Sand", "Coal",
              "Quartz", "Obsidian", "Prismarine", "TNT", "Sea Lantern"]
    disp = metrics.ConfusionMatrixDisplay(
        con_matrix,
        display_labels=labels)
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


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
