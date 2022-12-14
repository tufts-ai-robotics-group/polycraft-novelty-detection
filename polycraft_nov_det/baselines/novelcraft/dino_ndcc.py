from pathlib import Path

import numpy as np
import torch

from polycraft_nov_data.dataloader import novelcraft_dataloader
from polycraft_nov_data.image_transforms import VGGPreprocess

from polycraft_nov_det.baselines.novelcraft.ndcc import mahalanobis_metric
from polycraft_nov_det.detector import NoveltyDetector


class DINONDCCDetector(NoveltyDetector):
    def __init__(self, model, device="cpu"):
        super().__init__(device)
        self.model = model.eval().to(device)
        # init Gaussian params
        means, sigma2s = model.gaussian_params()
        self.means = means.cpu().detach().numpy()
        self.inv_sigma = np.diag(sigma2s.detach().cpu().numpy() ** -1)

    def novelty_score(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            outputs = self.model(data)
            outputs = outputs.detach().cpu().numpy().squeeze()
        distances = mahalanobis_metric(outputs, self.means, self.inv_sigma)
        nd_scores = np.min(distances, axis=1)
        return torch.Tensor(nd_scores)


if __name__ == "__main__":
    from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save

    valid_loader = novelcraft_dataloader("valid_norm", VGGPreprocess(), 64)
    test_loader = novelcraft_dataloader("test", VGGPreprocess(), 64)
    device = torch.device("cuda:1")
    model = torch.hub.load("tufts-ai-robotics-group/CCGaussian", "dino_ccg")
    output_folder = Path("models/dino_ndcc/")
    save_scores(DINONDCCDetector(model, device), output_folder, valid_loader, test_loader)
    eval_from_save(output_folder)
