import numpy as np

from polycraft_nov_det.ss_kmeans import ss_kmeans_plusplus


def test_ss_kmeans_plusplus():
    X = np.array([[10, 2], [10, 5], [10, 0]])
    y = np.array([1] * 3)
    X_unlabeled = np.array([[1, 2], [1, 4], [1, 0]])
    centers = ss_kmeans_plusplus(X, y, X_unlabeled, 1)
    # verify labeled center is the mean of the input
    assert np.all(centers[0] == np.mean(X, axis=0))
    # verify unlabeled center is in X_unlabeled
    assert np.any(np.all(centers[1] == X_unlabeled, axis=1))
