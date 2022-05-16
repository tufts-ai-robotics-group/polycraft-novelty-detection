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


if __name__ == "__main__":
    from pathlib import Path

    from polycraft_nov_det.baselines.eval_polycraft import save_scores, eval_from_save
    from polycraft_nov_det.model_utils import load_vgg_model

    device = torch.device("cuda:1")
    models = [
        load_vgg_model(Path("models/vgg/vgg_classifier_1000.pt", device)),
        load_vgg_model(Path("models/vgg/vgg_classifier_1000_2.pt", device)),
        load_vgg_model(Path("models/vgg/vgg_classifier_1000_3.pt", device)),
        load_vgg_model(Path("models/vgg/vgg_classifier_1000_4.pt", device)),
        load_vgg_model(Path("models/vgg/vgg_classifier_1000_5.pt", device)),
    ]
    output_folder = Path("models/vgg/eval_ensemble/")

    save_scores(EnsembleDetector(models, device), output_folder)
    eval_from_save(output_folder)
