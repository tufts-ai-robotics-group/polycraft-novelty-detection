from polycraft_nov_det.bootstrap import bootstrap_metric

import numpy as np


def _accuracy(y_pred, y_true):
    """
    helper function for polycraft_gcd
    calculates the accuracy of the predictions
    """
    return np.mean(y_pred == y_true)


def test_bootstrap_perfect():
    y_pred = np.zeros(20)
    y_true = np.zeros(20)

    metric_mean, ci_low, ci_high = bootstrap_metric(y_pred,
                                                    y_true,
                                                    _accuracy)
    assert metric_mean == 1
    assert ci_low == 1
    assert ci_high == 1


def test_bootstrap_multidim():
    y_pred = np.zeros((20, 2))
    y_true = np.ones((20, 2))

    metric_mean, ci_low, ci_high = bootstrap_metric(y_pred,
                                                    y_true,
                                                    _accuracy)

    assert metric_mean == 0
    assert ci_low == 0
    assert ci_high == 0


def test_bootstrap_ci():
    # small dataset with true rate of ~0.6
    y_pred_small = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    y_true_small = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])

    # initialize confidence interval
    prev_ci_low_diff = np.inf
    prev_ci_high_diff = np.inf

    for size_multiplier in [1, 10, 100]:
        y_pred = y_pred_small.repeat(size_multiplier)
        y_true = y_true_small.repeat(size_multiplier)

        metric_mean, ci_low, ci_high = bootstrap_metric(y_pred,
                                                        y_true,
                                                        _accuracy)

        # check that the confidence interval is getting smaller
        # as the dataset size increases
        ci_low_diff = np.abs(ci_low - metric_mean)
        ci_high_diff = np.abs(ci_high - metric_mean)

        assert ci_low_diff < prev_ci_low_diff
        assert ci_high_diff < prev_ci_high_diff

        prev_ci_low_diff = ci_low_diff
        prev_ci_high_diff = ci_high_diff
