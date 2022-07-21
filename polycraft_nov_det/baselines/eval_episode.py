from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

import polycraft_nov_data.episode_const as ep_const

from polycraft_nov_det.detector import NoveltyDetector


def save_scores(detector: NoveltyDetector, output_folder, test_loader):
    # collect scores, novelty labels with 1 as novel, and paths
    novel_scores = torch.Tensor([])
    paths = np.array([])
    for data, path in test_loader:
        novel_scores = torch.hstack([novel_scores, detector.novelty_score(data).cpu()])
        paths = np.hstack([paths, path])
    # output data
    folder_path = Path(output_folder)
    folder_path.mkdir(exist_ok=True, parents=True)
    torch.save(novel_scores, folder_path / "novel_scores.pt")
    np.save(folder_path / "paths.npy", paths)


def eval_from_save(output_folder):
    folder_path = Path(output_folder)
    novel_scores = torch.load(folder_path / "novel_scores.pt")
    paths = np.load(folder_path / "paths.npy")
    # for each novelty type produce list of episode scores in order of occurrence
    nov_type_to_scores = {}
    for novel_score, path in zip(novel_scores, paths):
        path = Path(path)
        nov_type, ep_num, image_name = path.parts
        ep_num = int(ep_num)
        frame_num = int(path.stem)
        # get list of episodes and pad if needed
        ep_scores = nov_type_to_scores.get(nov_type, [])
        ep_scores += [torch.tensor([])] * (ep_num - len(ep_scores) + 1)
        # pad list of scores and assign values
        if frame_num - len(ep_scores[ep_num]) + 1 > 0:
            ep_scores[ep_num] = torch.hstack(
                (ep_scores[ep_num], torch.zeros((frame_num - len(ep_scores[ep_num]) + 1,))))
        ep_scores[ep_num][frame_num] = novel_score
        nov_type_to_scores[nov_type] = ep_scores
    # visualize trials
    for nov_type, ep_scores in nov_type_to_scores.items():
        # remove last frame in each episode due to bug
        ep_scores = [frame_scores[:-1] for frame_scores in ep_scores]
        # reduce episode to max score from frames
        ep_scores = [torch.max(frame_scores) for frame_scores in ep_scores]
        # get max normal reconstruction error
        first_novel_ep = ep_const.TEST_CLASS_FIRST_NOVEL_EP[nov_type]
        max_norm = max(ep_scores[:first_novel_ep])
        # bar chart with lines for max normal reconstruction error and novelty split
        plt.figure(dpi=150)
        plt.bar(range(first_novel_ep), ep_scores[:first_novel_ep], color="red")
        plt.bar(range(first_novel_ep, len(ep_scores)), ep_scores[first_novel_ep:], color="green")
        plt.axhline(max_norm, color="black")
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
