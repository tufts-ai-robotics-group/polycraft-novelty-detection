import torch

from polycraft_nov_det.detector import NoveltyDetector


class EnsembleDetector(NoveltyDetector):
    def __init__(self, models, device="cpu"):
        super().__init__(device)
        self.models = [model.eval().to(device) for model in models]

    @torch.no_grad()
    def novelty_score(self, data):
        data = data.to(self.device)
        outputs = []
        for model in self.models:
            outputs += [torch.softmax(model(data), dim=1)]
        outputs = torch.stack(outputs)
        # selecting first element is only to discard argmax from torch.max
        return -1 * torch.max(torch.mean(outputs, dim=0), dim=-1)[0]
