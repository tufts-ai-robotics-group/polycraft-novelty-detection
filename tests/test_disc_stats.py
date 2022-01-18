import numpy as np

import polycraft_nov_det.eval.stats as stats


def test_assign():
    y_pred = np.array([0, 2, 0, 2, 2, 1, 1])
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    assert np.all(row_ind == np.array([0, 1, 2]))
    assert np.all(col_ind == np.array([0, 2, 1]))
    assert np.all(weight == np.array(
        [[1, 1, 0],
         [0, 0, 2],
         [1, 2, 0]]))


def test_accuracy():
    y_pred = np.array([0, 2, 0, 2, 2, 1, 1])
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    assert acc == 0.7142857142857143


def test_confusion():
    y_pred = np.array([0, 2, 0, 2, 2, 1, 1])
    y_true = np.array([0, 0, 1, 1, 1, 2, 2])
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    con = stats.cluster_confusion(row_ind, col_ind, weight)
    assert np.all(con == np.array(
        [[1., 1., 0.],
         [1., 2., 0.],
         [0., 0., 2.]]))
