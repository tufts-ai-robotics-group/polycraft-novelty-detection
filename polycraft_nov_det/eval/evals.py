from pathlib import Path

import torch
import numpy as np

from polycraft_nov_data.dataloader import novelcraft_dataloader
import polycraft_nov_data.novelcraft_const as n_const

from polycraft_nov_det.dino_trans import DINOTestTrans
import polycraft_nov_det.eval.stats as stats
from polycraft_nov_det.plot import plot_gcd_con_matrix, plot_gcd_CI
from polycraft_nov_det.ss_gmm import SSGMM
from polycraft_nov_det.ss_kmeans import SSKMeans
from polycraft_nov_det.bootstrap import bootstrap_metric
import matplotlib.pyplot as plt

def _accuracy_on_subset(num_norm, y_pred, y_true, category="normal"):
    """
    helper function for polycraft_gcd
    calculates the accuracy of the predictions on the normal and novel subsets
    """
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    con_mat = stats.cluster_confusion(row_ind, col_ind, weight)
    # clear rows so only looking at predictions with the desired true labels
    norm_con = np.copy(con_mat)
    norm_con[num_norm:] = 0
    novel_con = np.copy(con_mat)
    novel_con[:num_norm] = 0

    if category == "normal":
        norm_con = np.copy(con_mat)
        norm_con[num_norm:] = 0
        acc = float(np.diag(norm_con).sum()) / norm_con.sum()
    elif category == "novel":
        novel_con = np.copy(con_mat)
        novel_con[:num_norm] = 0
        acc = float(np.diag(novel_con).sum()) / novel_con.sum()
    else:
        raise ValueError("category must be 'normal' or 'novel'")

    return acc

@torch.no_grad()
def polycraft_gcd(model, label="GCD", device="cpu", embedding_ind=None, 
                  bootstrap=False, n_bootsraps=100):
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
        outputs = model(data)
        if embedding_ind is None:
            data_embeddings = outputs.detach().cpu().numpy()
        else:
            data_embeddings = outputs[embedding_ind].detach().cpu().numpy()
        labeled_embeddings = np.vstack((labeled_embeddings, data_embeddings))
        labeled_y = np.hstack((labeled_y, targets.cpu().numpy()))
    # collect unlabeled embeddings and labels
    embeddings = np.empty((0, 768))
    y_true = np.empty((0,))
    for data, targets in unlabeled_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        if embedding_ind is None:
            data_embeddings = outputs.detach().cpu().numpy()
        else:
            data_embeddings = outputs[embedding_ind].detach().cpu().numpy()
        embeddings = np.vstack((embeddings, data_embeddings))
        y_true = np.hstack((y_true, targets.cpu().numpy()))
    # apply SS clustering and output results
    for ss_method in ["KMeans", "GMM"]:
        if ss_method == "KMeans":
            # SS KMeans
            ss_est = SSKMeans(labeled_embeddings, labeled_y, 10).fit(
                embeddings)
            y_pred = ss_est.predict(embeddings)
        if ss_method == "GMM":
            # SS GMM
            ss_est = SSGMM(labeled_embeddings, labeled_y, embeddings, 10).fit(
                embeddings)
            y_pred = ss_est.predict(embeddings)
        # print results
        row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
        acc = stats.cluster_acc(row_ind, col_ind, weight)
        out_label = f"{label}_{ss_method}"
        print(f"{out_label}\n")
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
        plot_gcd_con_matrix(con_mat).savefig(fig_dir / f"{out_label}_con_mat.png")

        if bootstrap:
            # bootstrap
            # define metric function, which takes y_pred and y_true 
            # as arguments to stats.assign_clusters, and unpacks the returned values
            # to pass to stats.cluster_acc
            metric_func = lambda y_pred, y_true: \
                    stats.cluster_acc(*stats.assign_clusters(y_pred, y_true))
            mean, ci_low, ci_high = bootstrap_metric(y_pred, 
                                                    y_true, 
                                                    metric_func=metric_func, 
                                                    n_bootstraps=n_bootsraps)
            print(f"Bootstrap All: {mean:3f} ({ci_low:3f}, {ci_high:3f})")

            # get bootstrap results for normal and novel subsets
            metric_func = lambda y_pred, y_true: \
                    _accuracy_on_subset(num_norm, y_pred, y_true, category="normal")
            norm_mean, norm_ci_low, norm_ci_high = bootstrap_metric(y_pred,
                                                        y_true,
                                                        metric_func=metric_func,
                                                        n_bootstraps=n_bootsraps)
            print(f"Bootstrap Normal: {norm_mean:3f} ({norm_ci_low:3f}, {norm_ci_high:3f})")

            metric_func = lambda y_pred, y_true: \
                    _accuracy_on_subset(num_norm, y_pred, y_true, category="novel")
            novel_mean, novel_ci_low, novel_ci_high = bootstrap_metric(y_pred,
                                                        y_true,
                                                        metric_func=metric_func,
                                                        n_bootstraps=n_bootsraps)
            print(f"Bootstrap Novel: {novel_mean:3f} ({novel_ci_low:3f}, {novel_ci_high:3f})")
            
            plot_gcd_CI((mean, ci_low, ci_high,),
                        (norm_mean, norm_ci_low, norm_ci_high,),
                        (novel_mean, novel_ci_low, novel_ci_high,)).savefig(
                            fig_dir / f"{out_label}_CI.png")
    return acc

if __name__ == "__main__":
    from polycraft_nov_det.model_load import load_dino_block, load_dino_pretrained

    device = torch.device("cuda:0")

    polycraft_gcd(load_dino_pretrained(device), "DINO_SS_K-Means", device, 
                  embedding_ind=None, bootstrap=True, n_bootsraps=30)
    polycraft_gcd(load_dino_block("models/polycraft/GCD/block200.pt", device), "GCD_SS_K-Means",
                  device, embedding_ind=None, bootstrap=True, n_bootsraps=30)

    # polycraft_gcd(
    #     torch.hub.load(
    #         "tufts-ai-robotics-group/CCGaussian:main",
    #         "ccg_gcd",
    #         skip_validation=True,  # temp fix for torch bug
    #         trust_repo=True).to(device),
    #     "CCG_GCD_SS_K-Means", device, embedding_ind=1)
