import torch
import torch.nn as nn

from polycraft_nov_data.polycraft_dataloader import preprocess_image

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
from polycraft_nov_det.data_handler import read_image_csv


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

    def apply_model_and_compute_MSE(self, img):
        mse_loss = nn.MSELoss()
        x = preprocess_image(img, self.scale_factor, self.p_size)
        x =  torch.flatten(x, start_dim=0, end_dim=1)
        x_rec, z = self.model(x.float())
        rec_loss = mse_loss(x, x_rec)
        return rec_loss

    def check_for_novelty(self, img):
        print('We compute a score here, then we send it back!')
        img_rec_loss = self.apply_model_and_compute_MSE(img)
        # How exactly do we want to evaluate if an image is novel!?!?
        # res = ???
        return  # res


if __name__ == "__main__":
    # Test on csv from polycraft which includes the save_screen json!
    data, states = read_image_csv(
        path="novelty-lvl-1_2021-04-12-10-44-43_SENSE-SCREEN.csv", n_images=None, load_states=False)
    detector = VisualNoveltyDetector(
        state_dict_path='saved_statedict/LSA_polycraft_no_est_075_980.pt', scale_factor=0.75)
    detector.check_for_novelty(data[0])
