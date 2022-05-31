import torch
from torch.nn.functional import mse_loss

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_polycraft_model


class ReconstructDetector(NoveltyDetector):
    def __init__(self, model_path, device="cpu"):
        super().__init__(device)
        model = load_polycraft_model(model_path, device)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def novelty_score(self, data):
        data = data.to(self.device)
        r_data, embedding = self.model(data)

        return torch.max(mse_loss(data, r_data)).unsqueeze(0)


if __name__ == '__main__':

    from pathlib import Path

    from polycraft_nov_det.baselines.eval_polycraft import save_scores, eval_from_save

    output_parent = Path("models/vgg/eval_AE")
    output_folder = output_parent / Path("AE_patchwise")
    model_path = "models/polycraft/noisy/scale_1/patch_based/8000.pt"

    model_path = Path(model_path)

    save_scores(ReconstructDetector(model_path, device=torch.device("cuda:0")),
                output_folder, patch=True)
    eval_from_save(output_folder)
