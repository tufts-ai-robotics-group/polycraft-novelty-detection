from pathlib import Path

import torch

from polycraft_nov_data.dataloader import novelcraft_dataloader
from polycraft_nov_data.image_transforms import VGGPreprocess

from polycraft_nov_det.baselines.novelcraft.ndcc import NDCCDetector


if __name__ == "__main__":
    from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save

    valid_loader = novelcraft_dataloader("valid_norm", VGGPreprocess(), 64)
    test_loader = novelcraft_dataloader("test", VGGPreprocess(), 64)
    device = torch.device("cuda:1")
    model = torch.hub.load("tufts-ai-robotics-group/CCGaussian", "dino_ccg")
    output_folder = Path("models/dino_ndcc/")
    save_scores(NDCCDetector(model, device), output_folder, valid_loader, test_loader)
    eval_from_save(output_folder)
