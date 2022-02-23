import numpy as np

from polycraft_nov_det.data.cifar_loader import torch_cifar
import polycraft_nov_det.eval.stats as stats


def cifar10_self_supervised(model, device="cpu"):
    # get dataloader
    norm_targets, novel_targets, (_, _, test_loader) = torch_cifar(
        range(5), batch_size=128, include_novel=True, rot_loader="rotnet")
    # get model predictions
    model.eval()
    y_true = np.zeros((0,))
    y_pred = np.zeros((0,))
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        label_pred, unlabel_pred, feat = model(data)
        label_pred_max = np.argmax(label_pred.detach().cpu().numpy(), axis=1)
        # store outputs and targets
        y_true = np.hstack((y_true, targets.cpu().numpy()))
        y_pred = np.hstack((y_pred, label_pred_max))
    acc = stats.classification_acc(y_pred, y_true)
    print(acc)
    return acc


def cifar10_supervised(model, device="cpu"):
    # get dataloader
    norm_targets, novel_targets, (_, _, test_loader) = torch_cifar(
        range(5), batch_size=128)
    # get model predictions
    model.eval()
    y_true = np.zeros((0,))
    y_pred = np.zeros((0,))
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        label_pred, unlabel_pred, feat = model(data)
        label_pred_max = np.argmax(label_pred.detach().cpu().numpy(), axis=1)
        # store outputs and targets
        y_true = np.hstack((y_true, targets.cpu().numpy()))
        y_pred = np.hstack((y_pred, label_pred_max))
    acc = stats.classification_acc(y_pred, y_true)
    print(acc)
    return acc


def cifar10_clustering(model, device="cpu"):
    # get dataloader
    norm_targets, novel_targets, (_, _, test_loader) = torch_cifar(
        range(5), batch_size=128, include_novel=True)
    # get model predictions
    model.eval()
    y_true = np.zeros((0,))
    y_pred = np.zeros((0,))
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        label_pred, unlabel_pred, feat = model(data)
        unlabel_pred_max = np.argmax(unlabel_pred.detach().cpu().numpy(), axis=1)
        # select only unlabeled data
        norm_mask = targets < len(norm_targets)
        y_true = np.hstack((y_true, targets.cpu().numpy()[~norm_mask] - len(norm_targets)))
        y_pred = np.hstack((y_pred, unlabel_pred_max[~norm_mask]))
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    print(acc)
    print(stats.cluster_confusion(row_ind, col_ind, weight))
    return acc
