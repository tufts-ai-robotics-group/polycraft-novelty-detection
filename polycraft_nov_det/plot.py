import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA


def plot_con_matrix(con_matrix):
    disp = metrics.ConfusionMatrixDisplay(
        con_matrix, display_labels=["Novel", "Normal"])
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


def plot_gcd_con_matrix(con_matrix):
    labels = ["GS", "G1", "I1", "I2", "I3",
              "I4", "I5", "I6", "I7", "I8"]
    disp = metrics.ConfusionMatrixDisplay(
        con_matrix,
        display_labels=labels)
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


def plot_gcd_ci(all, normal, novel):
    """Plot confidence interval for GCD results

    Args:
        all (tuple): (mean, ci_low, ci_high) for all results
        normal (tuple): (mean, ci_low, ci_high) for normal results
        novel (tuple): (mean, ci_low, ci_high) for novel results
    Returns:
        plt.Figure: Figure with confidence interval plot
    """
    fig, ax = plt.subplots()
    x = np.arange(3)
    means = np.array([all[0], normal[0], novel[0]])
    ci_low = np.array([all[1], normal[1], novel[1]])
    ci_high = np.array([all[2], normal[2], novel[2]])
    ax.bar(x, means, yerr=[means - ci_low, ci_high - means], capsize=5,
           alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["All", "Normal", "Novel"])
    ax.set_ylabel("Accuracy")

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
        plt.scatter(embeddings_proj[is_target, 0],
                    embeddings_proj[is_target, 1], label=target)
    plt.legend()
    return fig
