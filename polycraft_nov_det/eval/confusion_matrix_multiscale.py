from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional
from torch.utils.data import DataLoader

from polycraft_nov_data import data_const as polycraft_const
from polycraft_nov_data.dataloader import polycraft_dataset_for_ms, polycraft_dataset
from polycraft_nov_data.dataset_transforms import folder_name_to_target_list
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.eval.plot as eval_plot
import polycraft_nov_det.eval.binary_classification_training_positions as bctp


def eval_polycraft_multiscale(model_paths, device="cpu"):

    # threshods determined for (non-noisy) single scales with cost fp = fn = 1
    thresholds = [0.003, 0.0061, 0.0144]  # not noisy
    #thresholds = [0.003, 0.0058, 0.0144]  # noisy
    fac = 0.9  # Multiply thresholds by this factor
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_path05 = Path(model_paths[0])
    model05 = model_utils.load_polycraft_model(model_path05, device).eval()
    model_path075 = Path(model_paths[1])
    model075 = model_utils.load_polycraft_model(model_path075, device).eval()
    model_path1 = Path(model_paths[2])
    model1 = model_utils.load_polycraft_model(model_path1, device).eval()

    _, valid_set05, test_set05 = polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.5, 
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_set075, test_set075 = polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.75, 
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_set1, test_set1 = polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=1, 
                                                            include_novel=True, 
                                                            shuffle=False)

    # get targets determined at runtime
    base_dataset = polycraft_dataset()                    
    loss_func2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]

    labels = []  # label 0 --> height, label 1 --> items, label 2 --> no novelty
    tp, fp, tn, fn = 0, 0, 0, 0
    # shapes of "patch array" for all scales.
    ipt_shapes = [[6, 7],
                  [9, 11],
                  [13, 15]]

    test_set = bctp.TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    with torch.no_grad():

        print('Iterate through images.')
        for i, samples in enumerate(test_loader):

            binary_loss = []

            for n, model in enumerate(all_models):

                patches = samples[n][0][0]

                patches = torch.flatten(patches, start_dim=0, end_dim=1)
                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE

                ipt_shape = ipt_shapes[n]

                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)

                loss2d = loss_func2d(x_rec, x)
                loss2d = torch.mean(loss2d, (1, 2, 3))  # avgd. loss per patch
                # Reshape loss values of flattened to "squared shape"
                loss2d = loss2d.reshape(ipt_shape[0], ipt_shape[1])
                # Interpolate smaller scales such that they match scale 1
                loss2d = functional.resize(loss2d.unsqueeze(0), (13, 15))
                ones = torch.ones_like(loss2d)
                zeros = torch.zeros_like(loss2d)
                loss2d = torch.where(loss2d > thresholds[n]*fac, ones, zeros)

                binary_loss.append(loss2d)  # Thresholded losses of all scales

            label = samples[0][1]

            if label == 0 or label == 1:
                label = True
            if label == 2:
                label = False

            labels.append(label)

            l05 = binary_loss[0]
            l075 = binary_loss[1]
            l1 = binary_loss[2]
            intersection1 = torch.where((l05 == 1) & (l075 == 1), ones, zeros)
            intersection2 = torch.where((l05 == 1) & (l1 == 1), ones, zeros)
            intersection3 = torch.where((l075 == 1) & (l1 == 1), ones, zeros)

            novelty_score = False

            if (torch.sum(intersection1) > 0 or torch.sum(intersection2) > 0
                                             or torch.sum(intersection3) > 0):
                novelty_score = True
   
            if novelty_score == True and label is True:
                tp += 1
            if novelty_score == True and label is False:
                fp += 1
            if novelty_score == False and label is False:
                tn += 1
            if novelty_score == False and label is True:
                fn += 1

    con_mat = np.array([[tp, fn],
                        [fp, tn]])

    del base_dataset
    return con_mat

      
if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    cm = eval_polycraft_multiscale(paths)
    print(cm)
    eval_plot.plot_con_matrix(cm).savefig(("con_matrix_ms.png"))

   