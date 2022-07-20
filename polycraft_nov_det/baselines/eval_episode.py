from pathlib import Path

from matplotlib import pyplot as plt
import torch

import polycraft_nov_data.episode_const as ep_const

from polycraft_nov_det.detector import NoveltyDetector


def save_scores(detector: NoveltyDetector, output_folder, test_loader):
    # collect scores, novelty labels with 1 as novel, and paths
    novel_scores = torch.Tensor([])
    paths = torch.Tensor([])
    for data, path in test_loader:
        novel_scores = torch.hstack([novel_scores, detector.novelty_score(data).cpu()])
        paths = torch.hstack([paths, path])
    # output data
    folder_path = Path(output_folder)
    folder_path.mkdir(exist_ok=True, parents=True)
    torch.save(novel_scores, folder_path / "novel_scores.pt")
    torch.save(folder_path / "paths.pt", paths)


def eval_from_save(output_folder):
    folder_path = Path(output_folder)
    novel_scores = torch.load(folder_path / "novel_scores.pt")
    paths = torch.load(folder_path / "paths.pt")
    # for each novelty type produce list of episode scores in order of occurrence
    nov_type_to_ep_scores = {}
    for novel_score, path in zip(novel_scores, paths):
        nov_type, ep_num, frame_num = Path(path).split()
        # assign episode max score from frames
        ep_scores = nov_type_to_ep_scores.get(nov_type, torch.zeros((100,)))
        ep_scores[ep_num] = max(ep_scores[ep_num], novel_score)
        nov_type_to_ep_scores[nov_type] = ep_scores
    # visualize trials
    for nov_type, ep_scores in nov_type_to_ep_scores.items():
        # TODO add lines for max/min pre/post novelty?
        plt.bar(range(len(ep_scores)), ep_scores)
        plt.title(nov_type)
        plt.savefig(output_folder / f"{nov_type}.png")
        plt.close()
    # CDT (correctly detected trials) = 0 FP and at least 1 TP
    # calc M1, average FN among CDTs
    # calc M2, % of trials that are CDTs
    # calc M2.1, % of trials with at least 1 FP
    return


if __name__ == "__main__":
    # TODO comparisons across model types
    pass
