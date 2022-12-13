from pathlib import Path

import numpy as np
import pandas as pd
import torch

from polycraft_nov_det.baselines.eval_novelcraft import detection_metrics


valid_nov_types = [
    "POGO_L00_T01_S01",
    "POGO_L01_T02_S01",
    "POGO_L02_T01_S01",
    "POGO_L02_T03_S01",
    "POGO_L04_T01_S01",
    "POGO_L04_T01_S02",
    "POGO_L04_T02_S01",
    "POGO_L04_T02_S02",
    "POGO_L04_T03_S01",
    "POGO_L04_T03_S02",
]

if __name__ == "__main__":
    base_folder = Path("models/episode/external")
    csv = pd.read_csv(base_folder / "TUFTSVIS_36M_AGENTS_TABLE_202212121132.csv")
    csv = csv[["Tournament_Name", "TOURNAMENT_TYPE", "Novelty_Detected", "Difficulty"]]
    # filter CSV to get visible novelties
    csv = csv.loc[csv["TOURNAMENT_TYPE"].isin(valid_nov_types)]
    # get novelty labels while fixing no novelty labels (POGO_L00_T01_S01)
    novel_true = torch.Tensor(np.logical_and(
        csv["Difficulty"].to_numpy() != "No Novelty",
        csv["TOURNAMENT_TYPE"].to_numpy() != "POGO_L00_T01_S01"))
    # only get binary value instead of actual scores
    novel_score = torch.Tensor(csv["Novelty_Detected"].to_numpy())
    # fix novelty only reported once per tournament
    tourney_name = csv["Tournament_Name"].to_numpy()
    last_name = ""
    for i in range(len(novel_score)):
        cur_name = tourney_name[i]
        if cur_name != last_name:
            last_name = cur_name
            novel_found = False
            print(f"Change at {i}")
        if novel_score[i] == 1:
            novel_found = True
        if novel_found:
            novel_score[i] = 1
    # detection metrics with normal weighting set so everything has weight 1
    tpr_95_thresh, prc_95_thresh = detection_metrics(
        base_folder, novel_true, novel_score, .33476, .95)[-2:]
