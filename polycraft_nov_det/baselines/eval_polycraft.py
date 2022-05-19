from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch

import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.dataloader import polycraft_dataloaders

from polycraft_nov_det.detector import NoveltyDetector


def save_scores(detector: NoveltyDetector, output_folder):
    (_, _, test_loader), class_to_idx = polycraft_dataloaders(
            include_novel=True, ret_class_to_idx=True, shuffle=False)
    normal_targets = torch.Tensor([class_to_idx[c] for c in data_const.NORMAL_CLASSES])
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # collect scores, novelty labels with 1 as novel, and targets
    novel_score = torch.Tensor([])
    novel_true = torch.Tensor([])
    targets = torch.Tensor([])
    for data, target in test_loader:
        novel_score = torch.hstack([novel_score, detector.novelty_score(data).cpu()])
        novel_true = torch.hstack([novel_true, (~torch.isin(target, normal_targets)).long()])
        targets = torch.hstack([targets, target])
    # convert targets to names
    classes = np.array([idx_to_class[target.item()] for target in targets])
    # output data
    folder_path = Path(output_folder)
    folder_path.mkdir(exist_ok=True, parents=True)
    torch.save(novel_score, folder_path / "novel_score.pt")
    torch.save(novel_true, folder_path / "novel_true.pt")
    np.save(folder_path / "classes.npy", classes)


def eval_from_save(output_folder):
    folder_path = Path(output_folder)
    novel_true = torch.load(folder_path / "novel_true.pt")
    novel_score = torch.load(folder_path / "novel_score.pt")
    # upsample normal data so it accounts for 3/4 of the weight, roughly the split of an episode
    # should affect PRC but not ROC
    norm_count = torch.sum(novel_true == 0)
    novel_count = torch.sum(novel_true == 1)
    weight = torch.ones_like(novel_score)
    weight[novel_true == 0] = 3 * novel_count / norm_count
    # ROC with 1 as novel target
    fpr, tpr, roc_threshs = metrics.roc_curve(novel_true, novel_score, sample_weight=weight)
    auroc = metrics.roc_auc_score(novel_true, novel_score, sample_weight=weight)
    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc).plot()
    plt.savefig(folder_path / "roc.png")
    plt.close()
    print(f"AUROC: {auroc}")
    # FPR at TPR 95%
    tpr_95_ind = np.argwhere(tpr >= .95)[0]
    print(f"FPR @ TPR {tpr[tpr_95_ind][0]}%: {fpr[tpr_95_ind][0]}")
    # PRC with 1 as novel target
    precision, recall, prc_threshs = metrics.precision_recall_curve(
        novel_true, novel_score, sample_weight=weight)
    av_p = metrics.average_precision_score(novel_true, novel_score, sample_weight=weight)
    metrics.PrecisionRecallDisplay(
        precision=precision, recall=recall, average_precision=av_p).plot()
    plt.savefig(folder_path / "prc.png")
    plt.close()
    print(f"Average Precision: {av_p}")
    # recall at precision 80%
    precision_80_ind = np.argwhere(precision >= .8)[0]
    print(f"Recall(TPR) @ Precision {precision[precision_80_ind][0]}%: " +
          f"{recall[precision_80_ind][0]}")
