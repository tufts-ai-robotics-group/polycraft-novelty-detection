from pathlib import Path

import numpy as np
import torch

from polycraft_nov_det.baselines.eval_episode import eval_from_save


if __name__ == "__main__":
    method_to_outputs = {
        "Autoencoder (Patch)": Path("models/episode/ae_patch/eval_patch/"),
        "NDCC": Path("models/episode/ndcc"),
    }
    method_to_nov_to_scores = {}
    for method, output_folder in method_to_outputs.items():
        print(f"Method: {method}")
        method_to_nov_to_scores[method] = eval_from_save(output_folder)
        print()
    # transforms to map both to [0, 1] on the validation set
    ae_trans = lambda x: x / .0065  # valid
    ndcc_trans = lambda x: (x - 110.9481) / (115.8725 - 110.9481)  # valid
    ae_nov_to_scores = method_to_nov_to_scores["Autoencoder (Patch)"]
    ndcc_nov_to_scores = method_to_nov_to_scores["NDCC"]
    nov_scores = torch.Tensor([])
    paths = np.array([])
    for nov in ae_nov_to_scores.keys():
        for ep in range(len(ae_nov_to_scores[nov])):
            cur_nov_scores = ae_trans(ae_nov_to_scores[nov][ep]) + \
                ndcc_trans(ndcc_nov_to_scores[nov][ep])
            cur_paths = [f"{nov}/{ep}/{i}" for i in range(len(ae_nov_to_scores[nov][ep]))]
            nov_scores = torch.hstack([nov_scores, cur_nov_scores])
            paths = np.hstack((paths, np.array(cur_paths)))
    torch.save(nov_scores, "models/episode/ndcc/ensemble_ndcc/novel_scores.pt")
    np.save("models/episode/ndcc/ensemble_ndcc/paths.npy", paths)
