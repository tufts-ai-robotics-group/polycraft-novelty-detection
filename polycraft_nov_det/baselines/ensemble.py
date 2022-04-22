import torch

from polycraft_nov_det.detector import NoveltyDetector


class EnsembleDetector(NoveltyDetector):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def novelty_score(self, data):
        outputs = []
        for model in self.models:
            outputs += [model(data)]
        outputs = torch.vstack(outputs)
        return torch.max(torch.mean(outputs, dim=0), dim=-1)
