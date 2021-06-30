import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch


# adapted from https://deeplizard.com/learn/video/0LhiS6yu2qQ
# - CNN Confusion Matrix with PyTorch - Neural Network Programming
def plot_confusion_matrix(cm, classes, t, scale, pool, cmap='BuPu'):
    fig3 = plt.figure()
    print('Confusion matrix, optimal threshold is ', t)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix, scale = %s, t = %s,  pooling = %s'
              % (str(scale), str(t), pool))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    color_thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > color_thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig3.savefig('conf.png', bbox_inches="tight")


def plot_roc(t_pos, f_pos, t_neg, f_neg):
    """
    Plot ROC curve based on previously determined false and true positives.
    """
    t_pos_rate = t_pos/(t_pos + f_neg)
    f_pos_rate = f_pos/(f_pos + t_neg)
    auc = -1 * np.trapz(t_pos_rate, f_pos_rate)

    plt.figure()
    plt.plot(f_pos_rate, t_pos_rate, linestyle='--', marker='o', color='darkorange',
             lw=2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC = %.2f' % (auc,))
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')


def plot_precision_recall(t_pos, f_pos, t_neg, f_neg):
    """
    Plot Precision Recall curve based on previously determined false and true
    positives and false and true negatives.
    """
    # If necessary, prevent division by zero
    allp = np.where((t_pos+f_pos) == 0, 1, (t_pos+f_pos))
    alln = np.where((t_pos+f_neg) == 0, 1, (t_pos+f_neg))

    prec = t_pos/allp
    prec = np.where(allp == 1, 1, prec)
    recall = t_pos/alln

    auc = metrics.auc(recall, prec)

    plt.figure()
    plt.plot(recall, prec, linestyle='--', marker='o', color='m', lw=2,
             clip_on=False)
    plt.plot([0, 1], [0.5, 0.5], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC = %.2f' % (auc,))
    plt.savefig('PR.png')


def confusion_stats(valid_loader, detector, thresholds, normal_targets):
    """
    Compute true positives, true negatives, false positives and false negatives
    (positive --> novel, negative --> non-novel)
    """
    t_pos, f_pos, t_neg, f_neg = np.zeros((4, len(thresholds)))
    with torch.no_grad():
        for data, targets in valid_loader:
            novel_pred = detector.is_novel(data, thresholds)
            novel_true = np.isin(targets.numpy(), normal_targets, invert=True)[np.newaxis]
            t_pos += np.sum(np.logical_and(novel_pred, novel_true), axis=1)
            f_pos += np.sum(np.logical_and(novel_pred, ~novel_true), axis=1)
            t_neg += np.sum(np.logical_and(~novel_pred, ~novel_true), axis=1)
            f_neg += np.sum(np.logical_and(~novel_pred, novel_true), axis=1)
    return t_pos, f_pos, t_neg, f_neg


def find_optimal_treshold(t_pos, f_pos, t_neg, f_neg, allts):
    """
    Compute optimal treshold based on true positives, true negatives, false
    positives and false negatives using cost function. The cost of false
    positives and false negatives are set to 1 (for now).
    """
    t_pos_rate = t_pos/(t_pos + f_neg)
    f_pos_rate = f_pos/(f_pos + t_neg)
    f_neg_rate = 1 - t_pos_rate
    total_cost = f_pos_rate + f_neg_rate
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = allts[optimal_idx]
    return optimal_threshold


def performance_evaluation():
    """
    Computes ROC and precision-recall curve for one (unseen) subset of non-novel
    and novel images. Then these can be used to compute an "optimal" threshold,
    which we evaluate on the other (unseen) subset of non-novel and novel
    images.
    """
    t_pos, f_pos, t_neg, f_neg = confusion_stats(
        valid_loader, model, allts, device, pooling
    )
    plot_roc(t_pos, f_pos, t_neg, f_neg, scale, pooling)
    plot_precision_recall(t_pos, t_neg, f_pos, f_neg, scale, pooling)

    thresh = find_optimal_treshold(t_pos, f_pos, t_neg, f_neg, allts)
    cm = compute_confusion_matrix(thresh, test_loader, model,
                                  device, pooling)

    classes = (['normal', 'novel'])
    plot_confusion_matrix(cm, classes, thresh, scale, pooling)
