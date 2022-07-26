from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
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
    nov_to_scores = {}
    for novel_score, path in zip(novel_scores, paths):
        path = Path(path)
        nov_type, ep_num, image_name = path.parts
        ep_num = int(ep_num)
        frame_num = int(path.stem)
        # get list of episodes and pad if needed
        ep_scores = nov_to_scores.get(nov_type, [])
        ep_scores += [torch.tensor([])] * (ep_num - len(ep_scores) + 1)
        # pad list of scores and assign values
        if frame_num - len(ep_scores[ep_num]) + 1 > 0:
            ep_scores[ep_num] = torch.hstack(
                (ep_scores[ep_num], torch.zeros((frame_num - len(ep_scores[ep_num]) + 1,))))
        ep_scores[ep_num][frame_num] = novel_score
        nov_to_scores[nov_type] = ep_scores

    # get list of max score for each episode
    nov_to_max_scores = {}
    for nov_type, ep_scores in nov_to_scores.items():
        # remove last frame in each episode due to bug
        ep_scores = [frame_scores[:-1] for frame_scores in ep_scores]
        # reduce episode to max score from frames
        max_scores = [torch.max(frame_scores) for frame_scores in ep_scores]
        nov_to_max_scores[nov_type] = max_scores

    # visualize trials
    for nov_type, max_scores in nov_to_max_scores.items():
        # get max normal reconstruction error
        first_novel_ep = ep_const.TEST_CLASS_FIRST_NOVEL_EP[nov_type]
        max_norm = max(max_scores[:first_novel_ep])
        # bar chart with lines for max normal reconstruction error and novelty split
        plt.figure(dpi=150)
        plt.bar(range(first_novel_ep), max_scores[:first_novel_ep], color="red")
        plt.bar(range(first_novel_ep, len(max_scores)), max_scores[first_novel_ep:], color="green")
        plt.axhline(max_norm, color="black")
        plt.title(nov_type)
        plt.savefig(output_folder / f"{nov_type}.png")
        plt.close()

    # aggregate trials and split into validation and test sets
    valid_scores = []
    valid_targets = []
    nov_to_test_tuple = {}
    rng = np.random.default_rng(42)
    for nov_type, max_scores in nov_to_max_scores.items():
        test_scores = []
        test_targets = []
        # randomly permute before splitting
        first_novel_ep = ep_const.TEST_CLASS_FIRST_NOVEL_EP[nov_type]
        norm_scores = list(rng.permuted(max_scores[:first_novel_ep]))
        novel_scores = list(rng.permuted(max_scores[first_novel_ep:]))
        # split data
        for i, scores in enumerate([norm_scores, novel_scores]):
            score_split_ind = len(scores) // 2
            for score_split, score_list, target_list in \
                    [(scores[score_split_ind:], valid_scores, valid_targets),
                     (scores[:score_split_ind], test_scores, test_targets)]:
                # add scores and targets to lists, with 0 normal and 1 novel
                score_list += score_split
                target_list += [i] * len(score_split)
        # combine test data and store according to novelty type
        nov_to_test_tuple[nov_type] = (np.array(test_scores), np.array(test_targets))

    # choose threshold from validation set to limit FPR < 5%
    fpr, tpr, thresholds = roc_curve(valid_targets, valid_scores)
    nov_thresh = thresholds[fpr < .05][-1]

    # bootstrap to generate test trials
    for nov_type, (test_scores, test_targets) in nov_to_test_tuple.items():
        norm_test_scores = test_scores[test_targets == 0]
        novel_test_scores = test_scores[test_targets == 1]
        n_norm = 15
        n_novel = 100 - n_norm
        n_trials = 1000
        bt_test_scores = np.hstack((
            rng.choice(norm_test_scores, (n_trials, n_norm)),
            rng.choice(novel_test_scores, (n_trials, n_novel)),
        ))
        # CDT (correctly detected trials) = 0 FP and at least 1 TP
        bt_pred = bt_test_scores >= nov_thresh
        cdt_mask = np.logical_and(np.all(~bt_pred[:, :n_norm], axis=1),
                                  np.any(bt_pred[:, n_norm:], axis=1))
        # calc M1, average FN among CDTs
        av_fn = 0
        for i, nov_preds in enumerate(bt_pred[cdt_mask, n_norm:]):
            cur_fn = 0
            for nov_pred in nov_preds:
                if not nov_pred:
                    cur_fn += 1
                else:
                    break
            av_fn = (cur_fn + av_fn * i) / (i + 1)
        # calc M2, % of trials that are CDTs
        percent_cdt = np.sum(cdt_mask) / n_trials * 100
        # calc M2.1, % of trials with at least 1 FP
        percent_fp = np.sum(np.any(bt_pred[:, :n_norm], axis=1)) / n_trials * 100
        # print results
        print(f"{nov_type} Results:")
        print(f"Av FN in CDTs: {av_fn}")
        print(f"Percent CDTs: {percent_cdt}")
        print(f"Percent FPs: {percent_fp}")
        print()
    return


if __name__ == "__main__":
    # TODO comparisons across model types
    pass
