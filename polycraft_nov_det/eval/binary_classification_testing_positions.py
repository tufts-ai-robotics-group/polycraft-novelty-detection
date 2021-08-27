from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from polycraft_nov_data import data_const as polycraft_const
import polycraft_nov_det.model_utils as model_utils
from polycraft_nov_data.dataloader import polycraft_dataset_for_ms, polycraft_dataset
import polycraft_nov_det.eval.plot as eval_plot
import polycraft_nov_det.eval.binary_classification_training_positions as bctp
import polycraft_nov_det.models.multiscale_classifier as ms_classifier


def find_optimal_threshold(P, N, FP, TP, FN, TN, allts):
    """
    Compute optimal treshold based on true positives, true negatives, false
    positives and false negatives using cost function.
    """
    frac_FP, frac_TP, frac_TN, frac_FN = FP/N, TP/P, TN/N, FN/P
    cost_FP, cost_TP, cost_TN, cost_FN = 1, 0, 0, 1
    total_cost = cost_FP * frac_FP + cost_TP * frac_TP + cost_TN * frac_TN + cost_FN * frac_FN
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = allts[optimal_idx]
    tp_opt = TP[optimal_idx]
    fn_opt = FN[optimal_idx]
    fp_opt = FP[optimal_idx]
    tn_opt = TN[optimal_idx]

    return tp_opt, fn_opt, fp_opt, tn_opt, optimal_threshold


def loss_vector_evaluation_pos(model_paths, allts, device):

    bc_path = 'models/polycraft/binary_classification/threshold_selection_pos_210.pt'
    classifier = ms_classifier.MultiscaleClassifierFeatureVector(9)
    classifier = model_utils.load_model(bc_path, classifier, device).eval()
    
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

    test_set = bctp.TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # get targets determined at runtime
    base_dataset = polycraft_dataset()                                         
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]

    tp, fp, tn, fn, pos, neg = 0, 0, 0, 0, 0, 0
    alltps = np.zeros(len(allts))
    allfps = np.zeros(len(allts))
    alltns = np.zeros(len(allts))
    allfns = np.zeros(len(allts))

    # shapes of "patch array" for all scales.
    ipt_shapes = [[6, 7],
                  [9, 11],
                  [13, 15]]

    i_ranges = []
    j_ranges = []
    i_ranges_norm = []
    j_ranges_norm = []

    for n, scale in enumerate(ipt_shapes):
        ipt_shape = ipt_shapes[n]
        i_ranges.append(np.array(range(0, ipt_shape[0])))
        j_ranges.append(np.array(range(0, ipt_shape[1])))
        i_ranges_norm.append(bctp.normalize_index_range(i_ranges[n]))
        j_ranges_norm.append(bctp.normalize_index_range(j_ranges[n]))

    with torch.no_grad():

        for i, samples in enumerate(test_loader):

            feature_vector = []
            i_idx_norm = []
            j_idx_norm = []

            for n, model in enumerate(all_models):

                patches = samples[n][0][0]
                patches = torch.flatten(patches, start_dim=0, end_dim=1)
                ipt_shape = ipt_shapes[n]
                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE

                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)

                loss1d = rec_loss2d(x_rec, x)
                loss1d = torch.mean(loss1d, (1, 2, 3))  # avgd. per patch
                maxloss = torch.max(loss1d)
                loss2d = loss1d.reshape(ipt_shape[0], ipt_shape[1]).cpu()
                max_idx = np.unravel_index(loss2d.argmax(), loss2d.shape)
                i_idx_norm.append(i_ranges_norm[n][max_idx[0]])
                j_idx_norm.append(j_ranges_norm[n][max_idx[1]])

                feature_vector.append(maxloss)

            feature_vector.append((i_idx_norm[0] - i_idx_norm[1])**2)
            feature_vector.append((i_idx_norm[0] - i_idx_norm[2])**2)
            feature_vector.append((i_idx_norm[1] - i_idx_norm[2])**2)
            feature_vector.append((j_idx_norm[0] - j_idx_norm[1])**2)
            feature_vector.append((j_idx_norm[0] - j_idx_norm[2])**2)
            feature_vector.append((j_idx_norm[1] - j_idx_norm[2])**2)

            label = samples[0][1]

            if label == 0 or label == 1:
                target = True
                pos += 1
            if label == 2:
                target = False
                neg += 1

            pred = classifier(torch.FloatTensor(feature_vector))#.to(device))

            for ii, t in enumerate(allts):

                if pred >= t and target is True:
                    alltps[ii] += 1
                if pred >= t and target is False:
                    allfps[ii] += 1
                if pred < t and target is False:
                    alltns[ii] += 1
                if pred < t and target is True:
                    allfns[ii] += 1

    tp, fn, fp, tn, t_opt = find_optimal_threshold(pos, neg, allfps, alltps,
                                                   allfns, alltns, allts)

    print('Optimal threshold', t_opt)

    con_mat = np.array([[tp, fn],
                   [fp, tn]])            
    del base_dataset
    return con_mat


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    allthreshs = np.round(np.linspace(0.1, 0.9, 21), 4)
    cm = loss_vector_evaluation_pos(paths, allthreshs, dev)
    print(cm)
    eval_plot.plot_con_matrix(cm).savefig(("con_matrix_bc_pos.png"))

  
