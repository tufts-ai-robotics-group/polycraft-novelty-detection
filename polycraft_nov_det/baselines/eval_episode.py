from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import torch

import polycraft_nov_data.episode_const as ep_const

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.baselines.eval_novelcraft import detection_metrics


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
    output_folder = Path(output_folder)
    novel_scores = torch.load(output_folder / "novel_scores.pt")
    paths = np.load(output_folder / "paths.npy")

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

    # drop ArenaBlockHard scores if included in data
    if "ArenaBlockHard" in nov_to_scores:
        nov_to_scores.pop("ArenaBlockHard")
    # filter out empty normal episode indices from only using test set
    if "normal" in nov_to_scores:
        nov_to_scores["normal"] = [scores for scores in nov_to_scores["normal"] if len(scores) > 0]
    # get list of max score for each episode
    nov_to_max_scores = {}
    for nov_type, ep_scores in nov_to_scores.items():
        # remove last frame in each episode due to bug
        ep_scores = [frame_scores[:-1] for frame_scores in ep_scores]
        # reduce episode to max score from frames
        max_scores = [torch.max(frame_scores) for frame_scores in ep_scores]
        nov_to_max_scores[nov_type] = max_scores

    tpr_95_thresh, prc_80_thresh = ep_detection_metrics(nov_to_max_scores, output_folder)
    print(f"Delay @ TPR: {av_delay_at_thresh(nov_to_scores, tpr_95_thresh)}")
    print(f"Delay @ Precision: {av_delay_at_thresh(nov_to_scores, prc_80_thresh)}")
    vis_trials(nov_to_max_scores, output_folder)
    return nov_to_scores


def ep_detection_metrics(nov_to_max_scores, output_folder):
    # construct tensors of novelty labels and scores
    novel_true = torch.Tensor([])
    novel_score = torch.Tensor([])
    for nov_type, max_scores in nov_to_max_scores.items():
        first_novel_ep = ep_const.TEST_CLASS_FIRST_NOVEL_EP[nov_type]
        cur_novel_true = torch.zeros(len(max_scores))
        cur_novel_true[first_novel_ep:] = 1
        novel_true = torch.hstack((novel_true, cur_novel_true))
        novel_score = torch.hstack((novel_score, torch.Tensor(max_scores)))
    # detection metrics with 15% normal 85% novel weighting
    tpr_95_thresh, prc_80_thresh = detection_metrics(
        output_folder, novel_true, novel_score, .15)[-2:]
    return tpr_95_thresh, prc_80_thresh


def vis_trials(nov_to_max_scores, output_folder):
    # visualize trials
    for nov_type, max_scores in nov_to_max_scores.items():
        # get max normal reconstruction error
        first_novel_ep = ep_const.TEST_CLASS_FIRST_NOVEL_EP[nov_type]
        if nov_type == "normal":  # normal will not have novel scores
            first_novel_ep = len(max_scores)
        max_norm = max(max_scores[:first_novel_ep])
        # bar chart with lines for max normal reconstruction error and novelty split
        plt.figure(dpi=150)
        plt.bar(range(first_novel_ep), max_scores[:first_novel_ep], color="red")
        if nov_type != "normal":  # normal will not have novel scores
            plt.bar(range(first_novel_ep, len(max_scores)), max_scores[first_novel_ep:],
                    color="green")
        plt.axhline(max_norm, color="black")
        plt.title(nov_type)
        plt.savefig(output_folder / f"{nov_type}.png")
        plt.close()


def bootstrap_metrics(nov_to_max_scores, output_folder):
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
    nov_thresh = .005  # TODO remove?
    # create figures for each statistic
    fig, (m1_ax, m2_ax, m21_ax) = plt.subplots(3, 1)
    m1_ax.set_ylabel("Av FN in CDTs")
    m2_ax.set_ylabel("Percentage CDTs")
    m21_ax.set_ylabel("Percentage FP > 0")
    m21_ax.set_xlabel("Threshold")
    for ax in (m1_ax, m2_ax, m21_ax):
        ax.tick_params(axis='x', labelsize=8)

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
        # examine range of values
        nov_thresh_range = .001
        num_interps = 10
        av_fns = np.zeros((num_interps,))
        percent_cdts = np.zeros((num_interps,))
        percent_fps = np.zeros((num_interps,))
        nov_threshs = np.linspace(
                nov_thresh - nov_thresh_range,
                nov_thresh + nov_thresh_range, num_interps)
        for i, temp_nov_thresh in enumerate(nov_threshs):
            # CDT (correctly detected trials) = 0 FP and at least 1 TP
            bt_pred = bt_test_scores >= temp_nov_thresh
            cdt_mask = np.logical_and(np.all(~bt_pred[:, :n_norm], axis=1),
                                      np.any(bt_pred[:, n_norm:], axis=1))
            # calc M1, average FN among CDTs
            av_fn = av_neg_before_pos(bt_pred[cdt_mask, n_norm:])
            # calc M2, % of trials that are CDTs
            percent_cdt = np.sum(cdt_mask) / n_trials * 100
            # calc M2.1, % of trials with at least 1 FP
            percent_fp = np.sum(np.any(bt_pred[:, :n_norm], axis=1)) / n_trials * 100
            # store results
            av_fns[i] = av_fn
            percent_cdts[i] = percent_cdt
            percent_fps[i] = percent_fp
        # plot results
        m1_ax.plot(nov_threshs, av_fns, label=nov_type)
        m2_ax.plot(nov_threshs, percent_cdts, label=nov_type)
        m21_ax.plot(nov_threshs, percent_fps, label=nov_type)
    m1_ax.legend()
    fig.savefig(output_folder / "bootstrap.png")
    return


def av_delay_at_thresh(nov_to_scores, thresh):
    # calculate average delay for each novelty, then average those
    thresh = float(thresh)
    delay = 0
    for nov_type, ep_scores in nov_to_scores.items():
        # get average false negatives before first positive for novel eps
        first_novel_ep = ep_const.TEST_CLASS_FIRST_NOVEL_EP[nov_type]
        ep_nov_preds = [frame_scores >= thresh for frame_scores in ep_scores[first_novel_ep:]]
        delay += av_neg_before_pos(ep_nov_preds)
    return delay / len(nov_to_scores.keys())


def av_neg_before_pos(eps_nov_preds):
    """Average number of negative predictions before first positive prediction

    Args:
        eps_nov_preds (list): For each episode, list of boolean novelty predictions

    Returns:
        float: Average number of negative predictions before first positive prediction
    """
    av_fn = 0
    for j, nov_preds in enumerate(eps_nov_preds):
        cur_fn = 0
        for nov_pred in nov_preds:
            if not nov_pred:
                cur_fn += 1
            else:
                break
        av_fn = (cur_fn + av_fn * j) / (j + 1)
    return av_fn


if __name__ == "__main__":
    method_to_outputs = {
        # "Autoencoder (Patch)": Path("models/episode/ae_patch/eval_patch/"),
        "JSON (2 step)": Path("models/episode/json/eval_json_2/"),
        "JSON (5 step)": Path("models/episode/json/eval_json_5/"),
        # "Multimodal (2 step)": Path("models/episode/multimodal/eval_multimodal/"),
        # "Multimodal (5 step)": Path("models/episode/multimodal/eval_multimodal_5/"),
    }
    for method, output_folder in method_to_outputs.items():
        print(f"Method: {method}")
        eval_from_save(output_folder)
        print()
