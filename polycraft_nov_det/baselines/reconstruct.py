import torch
from torch.nn.functional import mse_loss

from polycraft_nov_det.detector import NoveltyDetector


class ReconstructDetector(NoveltyDetector):
    def __init__(self, model, device="cpu"):
        super().__init__(device)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def novelty_score(self, data):
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        return torch.mean(mse_loss(data, r_data, reduction="none"),
                          (*range(1, data.dim()),))
