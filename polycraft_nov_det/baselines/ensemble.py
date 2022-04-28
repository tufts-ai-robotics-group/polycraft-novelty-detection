import torch

from polycraft_nov_det.detector import NoveltyDetector


class EnsembleDetector(NoveltyDetector):
    def __init__(self, models, device="cpu"):
        super().__init__(device)
        self.models = [model.eval().to(device) for model in models]

    def novelty_score(self, data):
        data = data.to(self.device)
        outputs = []
        for model in self.models:
            outputs += [model(data)]
        outputs = torch.vstack(outputs)
        return torch.max(torch.mean(outputs, dim=0), dim=-1)
