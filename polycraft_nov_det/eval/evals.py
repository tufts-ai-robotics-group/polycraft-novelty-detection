from pathlib import Path

import torch
import numpy as np

from polycraft_nov_data.dataloader import polycraft_dataloaders_gcd
import polycraft_nov_data.data_const as data_const

from polycraft_nov_det.data.cifar_loader import torch_cifar
from polycraft_nov_det.data.loader_trans import DINOTestTrans
import polycraft_nov_det.eval.stats as stats
from polycraft_nov_det.plot import plot_gcd_con_matrix
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


@torch.no_grad()
def polycraft_gcd(model, label="GCD", device="cpu"):
    # get dataloader
    batch_size = 128
    labeled_loader, unlabeled_loader = polycraft_dataloaders_gcd(
        DINOTestTrans(), batch_size)
    # collect labeled embeddings and labels
    labeled_embeddings = np.empty((0, 768))
    labeled_y = np.empty((0,))
    for data, targets in labeled_loader:
        data, targets = data.to(device), targets.to(device)
        data_embeddings = model(data).detach().cpu().numpy()
        labeled_embeddings = np.vstack((labeled_embeddings, data_embeddings))
        labeled_y = np.hstack((labeled_y, targets.cpu().numpy()))
    # collect unlabeled embeddings and labels
    embeddings = np.empty((0, 768))
    y_true = np.empty((0,))
    for data, targets in unlabeled_loader:
        data, targets = data.to(device), targets.to(device)
        data_embeddings = model(data).detach().cpu().numpy()
        embeddings = np.vstack((embeddings, data_embeddings))
        y_true = np.hstack((y_true, targets.cpu().numpy()))
    # SS KMeans
    ss_est = SSKMeans(labeled_embeddings, labeled_y, 10).fit(
        embeddings, torch.zeros_like(y_true))
    y_pred = ss_est.predict(embeddings)
    # print results
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    print(f"{label}\n")
    print(f"All: {acc}")
    # results for normal and novel subsets
    num_norm = len(data_const.NORMAL_CLASSES)
    norm_row_mask = row_ind < num_norm
    norm_weight = np.copy(weight)
    norm_weight[num_norm:] = 0
    norm_acc = stats.cluster_acc(row_ind[norm_row_mask], col_ind[norm_row_mask], norm_weight)
    print(f"Normal: {norm_acc}")
    novel_weight = np.copy(weight)
    novel_weight[:num_norm] = 0
    novel_acc = stats.cluster_acc(row_ind[~norm_row_mask], col_ind[~norm_row_mask], novel_weight)
    print(f"Novel: {novel_acc}")
    print("Confusion Matrix:")
    con_mat = stats.cluster_confusion(row_ind, col_ind, weight)
    print(con_mat)
    print()
    # confusion matrix visualization
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    plot_gcd_con_matrix(con_mat).savefig(fig_dir / f"{label}_con_mat.png")
    return acc


if __name__ == "__main__":
    from polycraft_nov_det.model_load import load_dino_block, load_dino_pretrained

    device = torch.device("cuda:1")
    polycraft_gcd(load_dino_pretrained(device), "DINO_SS_K-Means", device)
    polycraft_gcd(load_dino_block("models/polycraft/GCD/block200.pt", device), "GCD_SS_K-Means",
                  device)
