import numpy as np
import scipy.optimize as optimize


# cluster functions based on:
# https://github.com/k-han/AutoNovel/blob/5eda7e45898cf3fbcde4c34b9c14c743082abd94/utils/util.py#L19\


def assign_clusters(y_pred, y_true):
    """Calculate cluster assignments

    Args:
        y_true (np.array): true labels (N,)
        y_pred (np.array): predicted labels (N,)

    Returns:
        tuple: np.array of row indices and one of corresponding column indices giving the optimal
               assignment.
               np.array of weight matrix used for assignment.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    # add 1 because of zero indexing
    d = max(y_pred.max(), y_true.max()) + 1
    # compute weight matrix with row index as prediction and col index as true
    weight = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        weight[y_pred[i], y_true[i]] += 1
    # compute assignments
    row_ind, col_ind = optimize.linear_sum_assignment(weight, maximize=True)
    return row_ind, col_ind, weight


def cluster_acc(row_ind, col_ind, weight):
    """Compute clustering accuracy

    Args:
        row_ind (np.array): Predicted class index
        col_ind (np.array): True class index
        weight (np.array): Weight matrix used for assigning clusters

    Returns:
        float: accuracy, in [0,1]
    """
    return weight[row_ind, col_ind].sum() * 1.0 / weight.sum()


def cluster_confusion(row_ind, col_ind, weight):
    """Reorder weight matrix to get clustering confusion matrix

    Args:
        row_ind (np.array): Predicted class index
        col_ind (np.array): True class index
        weight (np.array): Weight matrix used for assigning clusters

    Returns:
        np.array: clustering confusion matrix
    """
    # add 1 because of zero indexing
    d = max(col_ind) + 1
    # reorder weights according to cluster assignments
    con_matrix = np.zeros((d, d))
    for i, j in zip(row_ind, col_ind):
        con_matrix[j] = weight[i]
    return con_matrix
