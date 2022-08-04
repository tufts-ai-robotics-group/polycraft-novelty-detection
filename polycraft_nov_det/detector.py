import torch


class NoveltyDetector:
    def __init__(self, device="cpu"):
        self.device = device

    @torch.no_grad()
    def novelty_score(self, data):
        return torch.zeros(data.shape[0])

    def is_novel(self, data, thresh):
        return self.novelty_score(data) > thresh
