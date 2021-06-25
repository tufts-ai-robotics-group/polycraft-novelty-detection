import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

from polycraft_nov_data.dataloader import polycraft_dataloaders
import polycraft_nov_det.model_utils as model_utils


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


def plot_roc(t_pos, f_pos, t_neg, f_neg, scale, pool):
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
    plt.title('AUC = %.2f, scale = %s, pooling = %s' % (auc, str(scale), pool))
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')


def plot_precision_recall(t_pos, f_pos, t_neg, f_neg, scale, pool):
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

    auc_pr = metrics.auc(recall, prec)

    plt.figure()
    plt.plot(recall, prec, linestyle='--', marker='o', color='m', lw=2,
             clip_on=False)
    plt.plot([0, 1], [0.5, 0.5], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC = %.2f, scale = %s, pooling = %s' %
              (auc_pr, str(scale), pool))
    plt.savefig('PR.png')


def classification_stats(valid_loader, model, allts, device, pooling,
                         scale):
    """
    Compute true positives, true negatives, false positives and false negatives
    (positive --> novel, negative --> non-novel)
    """
    loss_func2d = nn.MSELoss(reduction='none')
    t_pos = np.zeros(len(allts))
    f_pos = np.zeros(len(allts))
    t_neg = np.zeros(len(allts))
    f_neg = np.zeros(len(allts))
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            r_data, embedding = model(data)
            loss2d = loss_func2d(data, r_data)
            loss2d = torch.mean(loss2d, (1, 2, 3))  # averaged loss per patch

            if pooling == 'mean':
                # mean of all patch losses
                pooled_loss = torch.mean(loss2d).item()
            if pooling == 'max':
                # maximum of all patch losses
                pooled_loss = torch.max(loss2d).item()

            for ii, t in enumerate(allts):
                novelty_score = False

                if pooled_loss > t:
                    novelty_score = True

                if novelty_score is True:
                    f_pos[ii] += 1

                if novelty_score is False:
                    t_neg[ii] += 1

    plot_roc(t_pos, f_pos, t_neg, f_neg, scale, pooling)
    plot_precision_recall(t_pos, t_neg, f_pos, f_neg, scale, pooling)

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


def compute_confusion_matrix(t_opt, normal_test, model, device, pooling):
    """
    Compute confusion matrix for the second half of the available unseen data
    by using the optimal threshold we determined for the first half of the
    unseen data.
    """
    loss_func2d = nn.MSELoss(reduction='none')
    labels = []
    pred = []
    with torch.no_grad():
        for i, sample in enumerate(normal_test):
            patches = sample[0]
            x = patches.float().to(device)
            x.requires_grad = False
            x_rec, z = model(x)

            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))  # averaged loss per patch
            if pooling == 'mean':
                pooled_loss = torch.mean(loss2d).item()
            if pooling == 'max':
                pooled_loss = torch.max(loss2d).item()

            novelty_score = False
            if pooled_loss > t_opt:
                novelty_score = True
            pred.append(novelty_score)
    cm = confusion_matrix(labels, pred)  # Determine confusion matrix
    return cm


def performance_evaluation():
    """
    Computes ROC and precision-recall curve for one (unseen) subset of non-novel
    and novel images. Then these can be used to compute an "optimal" threshold,
    which we evaluate on the other (unseen) subset of non-novel and novel
    images.
    """
    model_directory = './models/polycraft/noisy/scale_1/8000.pt'
    scale = 1
    pool = 'max'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Use valid. set for threshold selection, test set for conf. matrix
    _, valid_loader, test_loader = polycraft_dataloaders(batch_size=1,
                                                         image_scale=scale, include_novel=True)
    # construct model
    model = model_utils.load_polycraft_model(model_directory, device)
    model.eval()
    #  max, 0.75, 0.5
    if scale == 0.75 or scale == 0.5:
        allts = np.round(np.linspace(0.003, 0.013, 30), 4)
        allts = np.append(0.000005, allts)
        allts = np.append(0.0005, allts)
        allts = np.append(allts, 0.015)
        allts = np.append(allts, 0.018)
        allts = np.append(allts, 0.02)
        allts = np.append(allts, 0.03)
        allts = np.append(allts, 0.04)
        allts = np.append(allts, 0.05)
        allts = np.append(allts, 0.06)
    #  max, 1
    if scale == 1:
        allts1 = np.round(np.linspace(0.003, 0.04, 40), 4)
        allts2 = np.round(np.linspace(0.04, 0.07, 20), 4)
        allts = np.append(allts1, allts2)

    t_pos, f_pos, t_neg, f_neg = classification_stats(
        valid_loader, model, allts, device, pool, scale
    )

    thresh = find_optimal_treshold(t_pos, f_pos, t_neg, f_neg, allts)
    cm = compute_confusion_matrix(thresh, test_loader, model,
                                  device, pool)

    classes = (['normal', 'novel'])
    plot_confusion_matrix(cm, classes, thresh, scale, pool)
