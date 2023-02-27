from polycraft_nov_det.bootstrap import bootstrap_metric

from sklearn.metrics import roc_auc_score
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def test_bootstrap():
    # case 1: perfect prediction
    y_pred = np.zeros(20)
    y_true = np.zeros(20)
    accuracy_func = lambda y_pred, y_true: np.mean(y_pred == y_true)

    metric_mean, ci_low, ci_high = bootstrap_metric(y_pred, 
                                                    y_true, 
                                                    accuracy_func)
    assert metric_mean == 1
    assert ci_low == 1
    assert ci_high == 1

    # case 2: 2D y_pred, opposite prediction
    y_pred = np.zeros((20, 2))
    y_true = np.ones((20, 2))

    metric_mean, ci_low, ci_high = bootstrap_metric(y_pred,
                                                    y_true,
                                                    accuracy_func)

    assert metric_mean == 0
    assert ci_low == 0
    assert ci_high == 0

    # case 3: large dataset
    # the metric should run fast and not crash
    y_pred = np.zeros(10**6)
    y_true = np.zeros(10**6)

    metric_mean, ci_low, ci_high = bootstrap_metric(y_pred,
                                                    y_true,
                                                    accuracy_func)
    
    assert metric_mean == 1
    assert ci_low == 1
    assert ci_high == 1

    # case 3: random prediction
    # small dataset with true rate of ~0.6
    y_pred_small = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    y_true_small = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])

    # roc_auc_score takes y_true as first argument and y_pred as second,
    # so we need to flip the order of the arguments
    metric_mean_s, ci_low_s, ci_high_s = bootstrap_metric(y_true_small,
                                                          y_pred_small,
                                                          roc_auc_score)
    logger.debug(f"Testing AUROC on small dataset with true rate of ~0.6")
    logger.debug(f"metric: roc_auc_score")
    logger.debug(f"metric_mean: {metric_mean_s:0.3f}")
    logger.debug(f"ci_low: {ci_low_s:0.3f}")
    logger.debug(f"ci_high: {ci_high_s:0.3f} \n")

    # medium dataset with true rate of ~0.6
    y_pred_med = y_pred_small.repeat(10)
    y_true_med = y_true_small.repeat(10)

    metric_mean_m, ci_low_m, ci_high_m = bootstrap_metric(y_true_med,
                                                          y_pred_med,
                                                          roc_auc_score)
    
    logger.debug(f"Testing AUROC on medium dataset with true rate of ~0.6")
    logger.debug(f"metric_mean: {metric_mean_m:0.3f}")
    logger.debug(f"ci_low: {ci_low_m:0.3f}")
    logger.debug(f"ci_high: {ci_high_m:0.3f} \n")

    # large dataset with true rate of ~0.6
    y_pred_large = y_pred_small.repeat(100)
    y_true_large = y_true_small.repeat(100)

    metric_mean_l, ci_low_l, ci_high_l = bootstrap_metric(y_true_large,
                                                          y_pred_large,
                                                          roc_auc_score)
    
    logger.debug(f"Testing AUROC on large dataset with true rate of ~0.6")
    logger.debug(f"metric_mean: {metric_mean_l:0.3f}")
    logger.debug(f"ci_low: {ci_low_l:0.3f}")
    logger.debug(f"ci_high: {ci_high_l:0.3f} \n")

    # according to the central limit theorem, the mean of the bootstrap
    # distribution should be close to the true mean, and the confidence
    # interval should be closer to the true confidence interval as the
    # sample size increases
    assert ci_low_s < ci_low_m < ci_low_l
    assert ci_high_s > ci_high_m > ci_high_l
