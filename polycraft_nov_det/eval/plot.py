from sklearn import metrics

from polycraft_nov_det.eval.stats import ratio


def plot_con_matrix(con_matrix):
    disp = metrics.ConfusionMatrixDisplay(con_matrix, display_labels=["Novel", "Normal"])
    disp = disp.plot(cmap="Blues", values_format=".0f")
    return disp.figure_


def plot_roc(t_pos, f_pos, t_neg, f_neg):
    """
    Plot ROC curve based on previously determined false and true positives.
    """
    f_pos_rate = ratio(f_pos, t_neg)
    t_pos_rate = ratio(t_pos, f_neg)
    roc_auc = metrics.auc(f_pos_rate, t_pos_rate)
    disp = metrics.RocCurveDisplay(fpr=f_pos_rate, tpr=t_pos_rate, roc_auc=roc_auc)
    disp = disp.plot()
    return disp.figure_


def plot_precision_recall(t_pos, f_pos, f_neg):
    """
    Plot Precision Recall curve based on previously determined false and true
    positives and false and true negatives.
    """
    prec = ratio(t_pos, f_pos)
    recall = ratio(t_pos, f_neg)
    disp = metrics.PrecisionRecallDisplay(prec, recall)
    disp = disp.plot()
    return disp.figure_
