from pathlib import Path

import numpy as np
import torch

from polycraft_nov_det.baselines.eval_episode import eval_from_save


if __name__ == "__main__":
    method_to_outputs = {
        "Autoencoder (Patch)": Path("models/episode/ae_patch/eval_patch/"),
        "JSON (2 step)": Path("models/episode/json/eval_json_2/"),
        "JSON (5 step)": Path("models/episode/json/eval_json_5/"),
    }
    method_to_nov_to_scores = {}
    for method, output_folder in method_to_outputs.items():
        print(f"Method: {method}")
        method_to_nov_to_scores[method] = eval_from_save(output_folder)
        print()
    # fuse patch and 5 step
    ae_factor = 10 / .004 * 4
    ae_nov_to_scores = method_to_nov_to_scores["Autoencoder (Patch)"]
    json_nov_to_scores = method_to_nov_to_scores["JSON (2 step)"]
    nov_scores = torch.Tensor([])
    paths = np.array([])
    for nov in ["tree_easy", "fence"]:
        for ep in range(len(ae_nov_to_scores[nov])):
            # fix issue with indexing
            if len(ae_nov_to_scores[nov][ep]) - len(json_nov_to_scores[nov][ep]) == 1:
                ae_nov_to_scores[nov][ep] = ae_nov_to_scores[nov][ep][1:]
            if len(ae_nov_to_scores[nov][ep]) - len(json_nov_to_scores[nov][ep]) == -1:
                json_nov_to_scores[nov][ep] = json_nov_to_scores[nov][ep][1:]
            cur_nov_scores = ae_factor * ae_nov_to_scores[nov][ep] + json_nov_to_scores[nov][ep]
            cur_paths = [f"{nov}/{ep}/{i}" for i in range(len(ae_nov_to_scores[nov][ep]))]
            nov_scores = torch.hstack([nov_scores, cur_nov_scores])
            paths = np.hstack((paths, np.array(cur_paths)))
    torch.save(nov_scores, "models/episode/multimodal/eval_multimodal/novel_scores.pt")
    np.save("models/episode/multimodal/eval_multimodal/paths.npy", paths)
