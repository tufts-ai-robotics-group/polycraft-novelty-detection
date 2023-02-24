

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
    metric_mean = 0.
    ci_low = 0.
    ci_high = 0.
    return (metric_mean, ci_low, ci_high)
