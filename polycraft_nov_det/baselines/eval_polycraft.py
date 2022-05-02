from sklearn.metrics import roc_curve
import torch

import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.dataloader import polycraft_dataloaders

from polycraft_nov_det.detector import NoveltyDetector


def roc_polycraft(detector: NoveltyDetector):
    # ROC with 1 as novel target
    (_, _, test_loader), class_to_idx = polycraft_dataloaders(include_novel=True,
                                                              ret_class_to_idx=True)
    normal_targets = torch.Tensor([class_to_idx[c] for c in data_const.NORMAL_CLASSES])
    novel_score = torch.Tensor([])
    novel_true = torch.Tensor([])
    for data, target in test_loader:
        novel_score = torch.hstack([novel_score, detector.novelty_score(data)])
        novel_true = torch.hstack([novel_true, (~torch.isin(target, normal_targets)).long()])
    novel_score = novel_score.detach().cpu().numpy()
    novel_true = novel_true.detach().cpu().numpy()
    return roc_curve(novel_true, novel_score)
