from pathlib import Path

import torch
import numpy as np

from polycraft_nov_data.dataloader import novelcraft_dataloader
import polycraft_nov_data.novelcraft_const as n_const

from polycraft_nov_det.dino_trans import DINOTestTrans
import polycraft_nov_det.eval.stats as stats
from polycraft_nov_det.plot import plot_gcd_con_matrix
from polycraft_nov_det.ss_kmeans import SSKMeans


@torch.no_grad()
def polycraft_gcd(model, label="GCD", device="cpu"):
    model.eval()
    # get dataloader
    batch_size = 128
    labeled_loader = novelcraft_dataloader("train", DINOTestTrans(), batch_size, True)
    unlabeled_loader = novelcraft_dataloader("valid", DINOTestTrans(), batch_size)
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
        embeddings, np.zeros_like(y_true))
    y_pred = ss_est.predict(embeddings)
    # print results
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    print(f"{label}\n")
    print(f"All: {acc}")
    # results for normal and novel subsets
    num_norm = len(n_const.NORMAL_CLASSES)
    con_mat = stats.cluster_confusion(row_ind, col_ind, weight)
    # clear rows so only looking at predictions with the desired true labels
    norm_con = np.copy(con_mat)
    norm_con[num_norm:] = 0
    norm_acc = float(np.diag(norm_con).sum()) / norm_con.sum()
    print(f"Normal: {norm_acc}")
    novel_con = np.copy(con_mat)
    novel_con[:num_norm] = 0
    novel_acc = float(np.diag(novel_con).sum()) / novel_con.sum()
    print(f"Novel: {novel_acc}")
    print("Confusion Matrix:")
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
