import torch
import json
import matplotlib.pyplot as plt

from polycraft_nov_data.polycraft_dataloader import preprocess_image

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
from polycraft_nov_det.novelty import load_ecdf, ReconstructionDet


class VisualNoveltyDetector:
    def __init__(self, state_dict_path, scale_factor):
        pc_input_shape = (3, 32, 32)  # color channels, height, width of one patch
        n_z = 110
        # construct model
        model = LSANet(pc_input_shape, n_z)
        # I compute it on a cpu, we might change this later!?
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
        model.eval()
        self.model = model
        self.scale_factor = scale_factor
        self.p_size = 32

    def apply_model_and_determine_novelty_judgement(self, img):
        # Construct Reconstruction error novelty detector class based on trained model
        # and previously computed (on non-novel dataset) and saved ecdf
        ecdf_no_novelty = load_ecdf(
            '../models/polycraft/saved_statedict_polycraft_scale_0_75/ecdf_LSA_polycraft_no_est_075_980.npy')
        rec_det = ReconstructionDet(self.model, ecdf_no_novelty)

        x = preprocess_image(img, self.scale_factor, self.p_size)
        x = torch.flatten(x, start_dim=0, end_dim=1).float()

        novelty_score = rec_det.is_novel(x, 0.99)

        return novelty_score

    def check_for_novelty(self, img):
        plt.imshow(img)
        plt.show()

        binary_novelty_score = self.apply_model_and_determine_novelty_judgement(img)

        # How exactly do we want to evaluate if an image is novel!?!?

        return binary_novelty_score.any()  # Return True if any patch was labeled as novel


if __name__ == "__main__":

    detector = VisualNoveltyDetector(
        state_dict_path='../models/polycraft/saved_statedict_polycraft_scale_0_75/LSA_polycraft_no_est_075_980.pt', scale_factor=0.75)

    with open("save_screen.json", 'r') as json_screen:
        img_json = json.load(json_screen)
        # convert to string in order to have the same SAVE_SCREEN format as received from the game
        img_json = str(img_json)

        score = detector.check_for_novelty(img_json)
