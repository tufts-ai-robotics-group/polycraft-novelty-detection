from pathlib import Path

from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import torch

import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.dataloader import polycraft_dataloaders

from polycraft_nov_det.detector import NoveltyDetector


def save_scores(detector: NoveltyDetector, output_folder):
    (_, _, test_loader), class_to_idx = polycraft_dataloaders(include_novel=True,
                                                              ret_class_to_idx=True)
    normal_targets = torch.Tensor([class_to_idx[c] for c in data_const.NORMAL_CLASSES])
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # collect scores, novelty labels with 1 as novel, and targets
    novel_score = torch.Tensor([])
    novel_true = torch.Tensor([])
    targets = torch.Tensor([])
    for data, target in test_loader:
        novel_score = torch.hstack([novel_score, detector.novelty_score(data).cpu()])
        novel_true = torch.hstack([novel_true, (~torch.isin(target, normal_targets)).long()])
        targets = torch.hstack([targets, targets])
    # convert targets to names
    classes = torch.Tensor([idx_to_class[target.item()] for target in targets])
    # output data
    folder_path = Path(output_folder)
    folder_path.mkdir(exist_ok=True)
    torch.save(novel_score, folder_path / "novel_score.pt")
    torch.save(novel_true, folder_path / "novel_true.pt")
    torch.save(classes, folder_path / "classes.pt")


def eval_from_save(output_folder):
    folder_path = Path(output_folder)
    novel_true = torch.load(folder_path / "novel_true.pt")
    novel_score = torch.load(folder_path / "novel_score.pt")
    # TODO need to upsample normal data for similar to typical episode
    # ROC with 1 as novel target
    fpr, tpr, roc_threshs = metrics.roc_curve(novel_true, novel_score)
    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig(folder_path / "roc.png")
    # PRC with 1 as novel target
    precision, recall, prc_threshs = metrics.precision_recall_curve(novel_true, novel_score)
    metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.savefig(folder_path / "prc.png")
    # TODO precision and FPR at TPR 95%
    # TODO TPR and FPR at precision 95%, may have to handle no precision near 95%
