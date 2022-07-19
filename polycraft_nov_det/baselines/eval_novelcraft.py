from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch

import polycraft_nov_data.novelcraft_const as nc_const

from polycraft_nov_det.detector import NoveltyDetector


def save_scores(detector: NoveltyDetector, output_folder, valid_loader, test_loader):
    # TODO either ensure loaders have batch size 1 or ensure code handles batch sizes > 1
    normal_targets = torch.Tensor([nc_const.ALL_CLASS_TO_IDX[c] for c in nc_const.NORMAL_CLASSES])
    idx_to_class = nc_const.ALL_IDX_TO_CLASS
    for split in ["valid", "test"]:
        loader = valid_loader if split == "valid" else test_loader
        # collect scores, novelty labels with 1 as novel, and targets
        novel_score = torch.Tensor([])
        novel_true = torch.Tensor([])
        targets = torch.Tensor([])
        for data, target in loader:
            novel_score = torch.hstack([novel_score, detector.novelty_score(data).cpu()])
            novel_true = torch.hstack([novel_true, (~torch.isin(target, normal_targets)).long()])
            targets = torch.hstack([targets, target])
        # convert targets to names
        classes = np.array([idx_to_class[target.item()] for target in targets])
        # output data
        folder_path = Path(output_folder)
        folder_path.mkdir(exist_ok=True, parents=True)
        torch.save(novel_score, folder_path / f"{split}_novel_score.pt")
        torch.save(novel_true, folder_path / f"{split}_novel_true.pt")
        np.save(folder_path / f"{split}_classes.npy", classes)


def eval_from_save(output_folder):
    folder_path = Path(output_folder)
    novel_true = torch.load(folder_path / "test_novel_true.pt")
    novel_score = torch.load(folder_path / "test_novel_score.pt")
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
    # TNR at TPR 95%
    tpr_95_ind = np.argwhere(tpr >= .95)[0]
    print(f"TNR @ TPR {tpr[tpr_95_ind][0]}%: {1 - fpr[tpr_95_ind][0]}")
    # PRC with 1 as novel target
    precision, recall, prc_threshs = metrics.precision_recall_curve(
        novel_true, novel_score, sample_weight=weight)
    auprc = metrics.auc(recall, precision)
    print(f"AUPRC: {auprc}")
    prc_threshs = np.hstack([prc_threshs, prc_threshs[-1] + 1e-4])  # extra thresh to match lens
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
    print('Prec 80 thresh', prc_threshs[precision_80_ind])
    # precision at TPR 95%
    prc_tpr_95_ind = np.argwhere(prc_threshs >= roc_threshs[tpr_95_ind])[0]
    print(f"Precision @ TPR {tpr[tpr_95_ind][0]}%: {precision[prc_tpr_95_ind][0]}")
    print('TPR 95 thresh', roc_threshs[tpr_95_ind])
    # TNR at precision 80%
    roc_precision_80_ind = np.argwhere(roc_threshs >= prc_threshs[precision_80_ind])[-1]
    print(f"TNR @ Precision {precision[precision_80_ind][0]}%: " +
          f"{1 - fpr[roc_precision_80_ind][0]}")
    print(f"TPR @ Precision {precision[precision_80_ind][0]}%: " +
          f"{tpr[roc_precision_80_ind][0]}")
    return fpr, tpr, auroc, precision, recall, av_p, auprc


if __name__ == "__main__":
    method_to_outputs = {
        "NDCC": Path("models/vgg/eval_ndcc/stanford_dogs_times_1e-1"),
        "ODIN": Path("models/vgg/eval_odin/t=1000_n=0.0000"),
        "Ensemble": Path("models/vgg/eval_ensemble/"),
        "One-Class SVM": Path("models/vgg/eval_ocsvm/nu=0.800000_gamm=0.000010"),
        "Autoencoder (Patch)": Path("models/polycraft/noisy/scale_1/patch_based/AE_patchwise"),
        "Autoencoder (Full image)":
        Path("models/polycraft/noisy/scale_1/fullimage_based/AE_fullimage")}
    for method, output_folder in method_to_outputs.items():
        print(f"Method: {method}")
        fpr, tpr, auroc, precision, recall, av_p, auprc = eval_from_save(output_folder)
        print()
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{method} (AUROC {auroc:.2%})")
        plt.figure(2)
        plt.plot(recall, precision, label=f"{method} (AUPRC {auprc:.2%})",
                 drawstyle="steps-post")
    # ROC figure
    plt.figure(1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("figures/roc.png")
    # PRC figure
    plt.figure(2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.savefig("figures/prc.png")
