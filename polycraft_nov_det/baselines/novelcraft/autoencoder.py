import numpy as np
import torch
from torch.nn.functional import mse_loss

import polycraft_nov_data.novelcraft_const as nc_const

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_autoencoder_model


class ReconstructDetector(NoveltyDetector):
    def __init__(self, model_path, input_shape, device="cpu"):
        super().__init__(device)
        model = load_autoencoder_model(model_path, input_shape, device)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def novelty_score(self, data):
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        return torch.mean(mse_loss(data, r_data, reduction="none"),
                          (*range(1, data.dim()),))


class ReconstructDetectorPatchBased(NoveltyDetector):
    def __init__(self, model_path, input_shape, device="cpu"):
        super().__init__(device)
        model = load_autoencoder_model(model_path, input_shape, device)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def novelty_score(self, data):
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        return torch.mean(mse_loss(data, r_data)).unsqueeze(0)


class AgentReconstructDetector(ReconstructDetectorPatchBased):
    def __init__(self, model_path, input_shape, device="cpu"):
        super().__init__(model_path, input_shape, device)

    def localization(self, data, scale):
        """Evaluate where something (potentially) novel appeared based on where the
           maximum reconstruction error (per patch) appears. Returns 0 if the error
           is highest in the leftmost column/third of the image, 1 if the error is highest in
           the central column/third of the image, 2 if it the error is highest in the
           rightmost column/third.

        Args:
            data (torch.tensor): Data to use as input to autoencoder and then pool

        Returns:
            column (int): 0 --> leftmost column, 1 --> central column,
            2 --> rightmost column
        """
        data = data.to(self.device)
        r_data, embedding = self.model(data)
        r_error = torch.mean(mse_loss(data, r_data, reduction="none"),
                             (*range(1, data.dim()),))
        # amount of patches per image width
        pw = nc_const.IMAGE_SHAPE[1]*scale/(nc_const.PATCH_SHAPE[1]//2) - 1
        # amount of patches per image height and width
        patches_shape = [int(data.shape[0]//pw), int(pw)]
        # reshape flattened patch error array to 2d
        r_error_per_patch = r_error.detach().cpu().numpy().reshape(patches_shape)
        # determine columns to which start new thirds of the image
        col_breaks = [patches_shape[1]//3, patches_shape[1] - patches_shape[1]//3]
        column = int(np.where(r_error_per_patch == np.max(r_error_per_patch))[1][0])
        out = 0
        if column >= col_breaks[0] and out < col_breaks[1]:
            out = 1
        elif column >= col_breaks[1]:
            out = 2
        return out


if __name__ == '__main__':
    from pathlib import Path

    from polycraft_nov_data.dataloader import collate_patches, novelcraft_dataloader
    from polycraft_nov_data.image_transforms import PatchTestPreprocess

    from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save

    # patch based AE
    output_parent = Path("models/polycraft/noisy/scale_1/patch_based")
    output_folder = output_parent / Path("AE_patchwise")
    model_path = "models/polycraft/noisy/scale_1/patch_based/8000.pt"

    model_path = Path(model_path)

    save_scores(ReconstructDetectorPatchBased(
            model_path, input_shape=(3, 32, 32), device=torch.device("cuda:0")),
        output_folder,
        novelcraft_dataloader("valid", PatchTestPreprocess(), 32, collate_fn=collate_patches),
        novelcraft_dataloader("test", PatchTestPreprocess(), 32, collate_fn=collate_patches),)
    eval_from_save(output_folder)

    # full image based AE
    output_parent = Path("models/polycraft/noisy/scale_1/fullimage_based")
    output_folder = output_parent / Path("AE_fullimage")
    model_path = "models/polycraft/noisy/scale_1/fullimage_based/8000.pt"

    model_path = Path(model_path)

    save_scores(ReconstructDetector(
            model_path, input_shape=(3, 256, 256), device=torch.device("cuda:0")),
        output_folder, quad_full_image=True)
    eval_from_save(output_folder)
