

import numpy as np

def bootstrap_metric(y_pred, y_true, metric_func):
    """Compute test set boostrapping of a metric

    Args:
        y_pred (np.array): Model predictions for some output y
        y_true (np.array): True value of output y
        metric_func (function): function with parameters (y_pred, y_true) returning a float metric

    Returns:
        tuple: metric_mean: float with bootstrapped mean of metric
               ci_low: Low value from 95% confidence interval
               ci_high: High value from 95% confidence interval
    """
    # set bootstrap sample size and random seed
    n_bootstraps = 200
    bootstrapped_scores = []
    rng = np.random.RandomState(123)
    idx = np.arange(y_pred.shape[0])

    # bootstrap
    for _ in range(n_bootstraps):
        indices = rng.choice(idx, size=y_pred.shape[0], replace=True)

        # sometimes bootstrapping will cause the metric to be undefined.
        # use try except to catch this and continue
        try:
            score = metric_func(y_pred[indices], y_true[indices])
        except:
            continue
        bootstrapped_scores.append(score)
    
    # compute mean and confidence interval
    metric_mean = np.mean(bootstrapped_scores)
    ci_low = np.percentile(bootstrapped_scores, 2.5)
    ci_high = np.percentile(bootstrapped_scores, 97.5)
    
    return (metric_mean, ci_low, ci_high)
