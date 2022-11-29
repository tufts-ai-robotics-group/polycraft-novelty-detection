from pathlib import Path

import numpy as np
import pandas
import torch

from polycraft_nov_data.dataset import EpisodeDataset


if __name__ == "__main__":
    base_path = Path("models/episode/json")
    win_sizes = ["2", "5"]
    test_dataset = EpisodeDataset("test")
    for win_size in win_sizes:
        csv_path = base_path / Path("per_step_errors_size_" + win_size + ".csv")
        out_folder = base_path / Path("eval_json_" + win_size)
        # parse CSV and split into objects
        csv_df = pandas.read_csv(csv_path, index_col=0)
        paths = np.array(csv_df["name"], dtype=str)
        novel_scores = torch.Tensor(csv_df["score"])
        # remove data normal from train and validation set
        test_mask = np.array([True] * len(paths))
        for i in range(len(paths)):
            cls_label, ep, _ = paths[i].split("/")
            if "normal" == cls_label:
                ep_ind = np.argwhere(test_dataset.ep_to_split == cls_label + "/" + ep)
                split_label = test_dataset.ep_to_split[ep_ind[0, 0], 1]
                if split_label != "test":
                    test_mask[i] = False
        paths = paths[test_mask]
        novel_scores = novel_scores[test_mask]
        # save objects
        np.save(out_folder / "paths.npy", paths)
        torch.save(novel_scores, out_folder / "novel_scores.pt")
