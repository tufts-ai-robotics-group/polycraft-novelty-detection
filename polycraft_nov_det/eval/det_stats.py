import numpy as np
import torch


def confusion_stats(valid_loader, detector, thresholds, normal_targets, pool_batches):
    """Compute confusion stats for multiple thresholds, with positives as novel

    Args:
        valid_loader (DataLoader): Validation set DataLoader with normal and novel data
        detector (ReconstructionDet): Detector to compute statistics with
        thresholds (np.ndarray): Set of detector thresholds to evaluate
        normal_targets (list): Set of targets to use as normal
        pool_batches (bool): Whether to used pooled detection on DataLoader batches

    Returns:
        tuple: 4 np.ndarray of length (N), containing following stats per threshold:
               true positives, false positives, true negatives, false negatives
    """
    t_pos, f_pos, t_neg, f_neg = np.zeros((4, len(thresholds)))
    with torch.no_grad():
        for data, targets in valid_loader:
            if pool_batches:
                # extra axis added to match non-pooled shape for sums
                novel_pred = detector.is_novel_pooled(data, thresholds)[:, np.newaxis]
            else:
                novel_pred = detector.is_novel(data, thresholds)
            novel_true = np.isin(targets.numpy(), normal_targets, invert=True)[np.newaxis]
            t_pos += np.sum(np.logical_and(novel_pred, novel_true), axis=1)
            f_pos += np.sum(np.logical_and(novel_pred, ~novel_true), axis=1)
            t_neg += np.sum(np.logical_and(~novel_pred, ~novel_true), axis=1)
            f_neg += np.sum(np.logical_and(~novel_pred, novel_true), axis=1)
    return t_pos, f_pos, t_neg, f_neg


def optimal_index(t_pos, f_pos, t_neg, f_neg):
    """Get optimal threshold index from confusion stats

    Args:
        t_pos (np.ndarray): (N) true positives per threshold
        f_pos (np.ndarray): (N) false positives per threshold
        t_neg (np.ndarray): (N) true negatives per threshold
        f_neg (np.ndarray): (N) false negatives per threshold

    Returns:
        int: Index of threshold minimizing false positives and negatives
    """
    f_pos_rate = ratio(f_pos, t_neg)
    f_neg_rate = ratio(f_neg, t_pos)
    total_cost = f_pos_rate + f_neg_rate
    return np.argmin(total_cost)


def optimal_con_matrix(t_pos, f_pos, t_neg, f_neg):
    """Get optimal threshold index from confusion stats

    Args:
        t_pos (np.ndarray): (N) true positives per threshold
        f_pos (np.ndarray): (N) false positives per threshold
        t_neg (np.ndarray): (N) true negatives per threshold
        f_neg (np.ndarray): (N) false negatives per threshold

    Returns:
        np.ndarray: Confusion matrix of threshold minimizing false positives and negatives
    """
    opt_ind = optimal_index(t_pos, f_pos, t_neg, f_neg)
    return np.array([[t_pos[opt_ind], f_neg[opt_ind]],
                     [f_pos[opt_ind], t_neg[opt_ind]]])


def ratio(a, b):
    # safely calculate a/(a + b) with a, b > 0, returning 0 for a + b = 0
    denom = a + b
    denom[denom == 0] = 1
    return a/denom
