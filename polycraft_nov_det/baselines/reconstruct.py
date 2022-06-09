import torch
from torch.nn.functional import mse_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_polycraft_model


class ReconstructDetector(NoveltyDetector):
    def __init__(self, model_path, ipt_shape, device="cpu"):
        super().__init__(device)
        model = load_polycraft_model(model_path, ipt_shape, device)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def novelty_score(self, data):
        
        data = data.to(self.device)
        #print(data.shape)
        r_data, embedding = self.model(data)
        #print(torch.mean(mse_loss(data, r_data, reduction="none"),
                         # (*range(1, data.dim()),)).shape)
        return torch.mean(mse_loss(data, r_data, reduction="none"),
                          (*range(1, data.dim()),))
    

class ReconstructDetectorPatchBased(NoveltyDetector):
    def __init__(self, model_path, ipt_shape, device="cpu"):
        super().__init__(device)
        model = load_polycraft_model(model_path, ipt_shape, device)
        self.model = model.eval().to(device)

    @torch.no_grad()
    def novelty_score(self, data):
        data = data.to(self.device)
        #print(data.shape)
        r_data, embedding = self.model(data)

        return torch.mean(mse_loss(data, r_data)).unsqueeze(0)


if __name__ == '__main__':

    from pathlib import Path

    from polycraft_nov_det.baselines.eval_polycraft import save_scores, eval_from_save
    
    # patch based AE
    output_parent = Path("models/polycraft/noisy/scale_1/patch_based")
    output_folder = output_parent / Path("AE_patchwise")
    model_path = "models/polycraft/noisy/scale_1/patch_based/8000.pt"

    model_path = Path(model_path)

    save_scores(ReconstructDetectorPatchBased(model_path, ipt_shape=(3, 32, 32), device=torch.device("cuda:0")), 
                output_folder, patch=True)
    eval_from_save(output_folder)  
    
    # full image based AE
    output_parent = Path("models/polycraft/noisy/scale_1/fullimage_based")
    output_folder = output_parent / Path("AE_fullimage")
    model_path = "models/polycraft/noisy/scale_1/fullimage_based/8000.pt"

    model_path = Path(model_path)
    
    save_scores(ReconstructDetector(model_path, ipt_shape=(3, 256, 256), device=torch.device("cuda:0")), 
                output_folder, quad_full_image=True)
    eval_from_save(output_folder)  