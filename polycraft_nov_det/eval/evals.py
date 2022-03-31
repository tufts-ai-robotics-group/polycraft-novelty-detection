import numpy as np

from polycraft_nov_det.data.cifar_loader import torch_cifar
import polycraft_nov_det.eval.stats as stats
from polycraft_nov_det.ss_kmeans import SSKMeans


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


def cifar10_autonovel(model, device="cpu"):
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
        y_true = np.hstack((y_true, targets.cpu().numpy()[~norm_mask.cpu()] - len(norm_targets)))
        y_pred = np.hstack((y_pred, unlabel_pred_max[~norm_mask.cpu()]))
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    print(acc)
    print(stats.cluster_confusion(row_ind, col_ind, weight))
    return acc


def cifar10_gcd(model, device="cpu"):
    # get dataloader
    batch_size = 128
    norm_targets, novel_targets, (train_loader, _, _) = torch_cifar(
            range(5), batch_size, include_novel=True, rot_loader=None)
    # collect embeddings and labels
    embeddings = np.empty((0, 768))
    y_true = np.empty((0,))
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        data_embeddings = model(data).detach().cpu().numpy()
        embeddings = np.vstack((embeddings, data_embeddings))
        y_true = np.hstack((y_true, targets.cpu().numpy()))
    # SS KMeans
    norm_mask = y_true < len(norm_targets)
    ss_est = SSKMeans(embeddings[norm_mask], y_true[norm_mask], 10).fit(
        embeddings[~norm_mask], y_true[~norm_mask])
    y_pred = ss_est.predict(embeddings)
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    print(acc)
    print(stats.cluster_confusion(row_ind, col_ind, weight))
    return acc
