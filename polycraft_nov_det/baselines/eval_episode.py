from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch

import polycraft_nov_data.episode_const as ep_const

from polycraft_nov_det.detector import NoveltyDetector


def save_scores(detector: NoveltyDetector, output_folder, test_loader):
    # collect scores, novelty labels with 1 as novel, and paths
    novel_score = torch.Tensor([])
    novel_true = torch.Tensor([])
    paths = torch.Tensor([])
    for data, path in test_loader:
        novel_score = torch.hstack([novel_score, detector.novelty_score(data).cpu()])
        novel_true = torch.hstack([novel_true,
                                   (~torch.isin(str(path.root), ep_const.NORMAL_CLASSES)).long()])
        paths = torch.hstack([paths, path])
    # output data
    folder_path = Path(output_folder)
    folder_path.mkdir(exist_ok=True, parents=True)
    torch.save(novel_score, folder_path / "novel_score.pt")
    torch.save(novel_true, folder_path / "novel_true.pt")
    torch.save(folder_path / "paths.pt", paths)


def eval_from_save(output_folder):
    folder_path = Path(output_folder)
    novel_true = torch.load(folder_path / "test_novel_true.pt")
    novel_score = torch.load(folder_path / "test_novel_score.pt")
    paths = torch.load(folder_path / "paths.pt")
    # for each class produce list of episode scores in order of occurrence
    # CDT (correctly detected trials) = 0 FP and at least 1 TP
    # calc M1, average FN among CDTs
    # calc M2, % of trials that are CDTs
    # calc M2.1, % of trials with at least 1 FP
    return


if __name__ == "__main__":
    # TODO comparisons across model types
    pass
