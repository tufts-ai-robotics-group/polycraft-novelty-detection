from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import functional
import matplotlib.pyplot as plt
import os

from polycraft_nov_data import data_const as polycraft_const
import polycraft_nov_det.model_utils as model_utils
from polycraft_nov_data.dataloader import polycraft_dataset_for_ms, polycraft_dataset
import polycraft_nov_det.eval.plot as eval_plot
import polycraft_nov_det.eval.binary_classification_training_positions as bctp 
import polycraft_nov_data.image_transforms as image_transforms
import polycraft_nov_det.models.multiscale_classifier as ms_classifier


def find_optimal_threshold(P, N, FP, TP, FN, TN, allts):
    """
    Compute optimal treshold based on true positives, true negatives, false
    positives and false negatives using cost function.
    """
    frac_FP, frac_TP, frac_TN, frac_FN = FP/N, TP/P, TN/N, FN/P
    cost_FP, cost_TP, cost_TN, cost_FN = 1, 0, 0, 2
    total_cost = (cost_FP * frac_FP + cost_TP * frac_TP + 
                  cost_TN * frac_TN + cost_FN * frac_FN)
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = allts[optimal_idx]
    tp_opt = TP[optimal_idx]
    fn_opt = FN[optimal_idx]
    fp_opt = FP[optimal_idx]
    tn_opt = TN[optimal_idx]

    return tp_opt, fn_opt, fp_opt, tn_opt, optimal_threshold


def plot_fp_fn_examples(testloader, allmodels, detector, topt, device):
   
    if not os.path.exists('polycraft_nov_det/eval/evaluation/'):
        os.makedirs('polycraft_nov_det/eval/evaluation/')
    
    rec_loss2d = nn.MSELoss(reduction='none')
    # shapes of "patch array" for all scales.
    ipt_shapes = [[6, 7],
                  [9, 11],
                  [13, 15]]
    for i, samples in enumerate(testloader):

        loss_arrays = ()
        loss2d_allscales = []
        x_rec_allscales = []

        with torch.no_grad():

            for n, model in enumerate(allmodels):

                ipt_shape = ipt_shapes[n]
                patches = samples[n][0][0]
                patches = torch.flatten(patches, start_dim=0, end_dim=1)

                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE

                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)

                loss2d = rec_loss2d(x_rec, x)
                # Pixelweise loss averaged across rgb channels
                loss2d_ = torch.mean(loss2d, 1, True)
                loss2d = torch.mean(loss2d, (1, 2, 3))  # avgd. per patch
                # Reshape loss values from flattened to "squared shape"
                loss2d = loss2d.reshape(1, 1, ipt_shape[0], ipt_shape[1])
                # Interpolate smaller scales such that they match scale 1
                loss2d = functional.resize(loss2d, (13, 15))
                loss_arrays = loss_arrays + (loss2d,)

                if n == 1:
                    # Plot x for scale 0.75
                    x_scale075 = x

                # Plot reconstructions and pixelwise loss for all scales
                loss2d_allscales.append(loss2d_)
                x_rec_allscales.append(x_rec)

            label = samples[0][1]

            if label == 0 or label == 1:
                target = True

            if label == 2:
                target = False

            pred = detector(loss_arrays)

            if pred >= topt and target is False:
                problempatch = x_scale075
                problempatch_recs = x_rec_allscales
                problempatch_losses = loss2d_allscales
                descrip = 'FP'
                print(descrip)
                plot_patches(problempatch, problempatch_recs,
                             problempatch_losses, descrip, i, device)

            if pred < topt and target is True:
                problempatch = x_scale075
                problempatch_recs = x_rec_allscales
                problempatch_losses = loss2d_allscales
                descrip = 'FN'
                print(descrip)
                plot_patches(problempatch, problempatch_recs,
                             problempatch_losses, descrip, i, device)

    return


def compute_mean_filtering_mask(n_c, img_shape, ipt_shape, img_dim, device):

    mask = torch.ones(img_shape)
    mask = image_transforms.ToPatches((n_c, 32, 32))(mask)
    mask = mask.reshape(ipt_shape)
    mask = torch.flatten(mask, start_dim=1, end_dim=2)
    mask = mask.contiguous().view(1, n_c*32*32, -1)
    mask = F.fold(
        mask, output_size=img_dim, kernel_size=32, stride=16)
    zeros = torch.zeros_like(mask)
    ones = torch.ones_like(mask)
    mask = torch.where(mask == zeros, ones, mask)

    return mask.to(device)


def compute_image_from_patches(n_c, patches, mask, ipt_shape, img_dim):

    patches = patches.reshape(ipt_shape)
    patches = patches.permute(0, 3, 1, 2, 4, 5)
    patches = patches.contiguous().view(1, n_c, -1, 32*32)
    patches = patches.permute(0, 1, 3, 2)
    patches = patches.contiguous().view(1, n_c*32*32, -1)
    img = F.fold(
        patches, output_size=img_dim, kernel_size=32, stride=16)
    img = img/mask

    return img


def plot_patches(patch_x, patch_x_recs, patch_losses, d, idx, device):

    scale = 0.75
    _, ih, iw = polycraft_const.IMAGE_SHAPE
    _, ph, pw = polycraft_const.PATCH_SHAPE
    image_shape3 = (3, int((ih - 22)*scale), int(iw*scale))

    ipt_shapes1 = [[1, 6, 7, 1, 32, 32],
                  [1, 9, 11, 1, 32, 32],
                  [1, 13, 15, 1, 32, 32]]
    
    ipt_shapes3 = [[1, 6, 7, 3, 32, 32],
                  [1, 9, 11, 3, 32, 32],
                  [1, 13, 15, 3, 32, 32]]

    scales = [0.5, 0.75, 1]
    image_dim = (int((ih - 22)*scale), int(iw*scale))

    mask_3c = compute_mean_filtering_mask(3, image_shape3, ipt_shapes3[1], 
                                          image_dim, device)

    image = compute_image_from_patches(3, patch_x, mask_3c, ipt_shapes3[1],
                                       image_dim)
    image = functional.resize(image, (ih - 22, iw))
    image_rec = compute_image_from_patches(3, patch_x_recs[1], mask_3c, 
                                           ipt_shapes3[1], image_dim)
    image_rec = functional.resize(image_rec, (ih - 22, iw))

    loss_image_all = []
    rec_image_all = []

    for i in range(3):  # Iterate over rec. and pixelwise loss of all scales
        image_dim = (int((ih - 22)*scales[i]), int(iw*scales[i]))
        image_shape1 = (1, int((ih - 22)*scales[i]), int(iw*scales[i]))
        mask_1c = compute_mean_filtering_mask(1, image_shape1, ipt_shapes1[i],
                                              image_dim, device)
        loss_image = compute_image_from_patches(1, patch_losses[i], mask_1c,
                                                ipt_shapes1[i], image_dim)
        loss_image_all.append(functional.resize(loss_image, (ih - 22, iw)))

        image_shape3 = (3, int((ih - 22)*scales[i]), int(iw*scales[i]))
        mask_3c = compute_mean_filtering_mask(3, image_shape3, ipt_shapes3[i],
                                              image_dim, device)
        rec_image = compute_image_from_patches(3, patch_x_recs[i], mask_3c,
                                                ipt_shapes3[i], image_dim)
        rec_image_all.append(functional.resize(rec_image, (ih - 22, iw)))

    fig = plt.figure(figsize=(20, 15), dpi=700)
    fig.suptitle(d)
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.title.set_text('x')
    ax1.imshow(np.transpose((image[0]).detach().cpu().numpy(),
                            (1, 2, 0))[:216, :])

    ax2 = fig.add_subplot(3, 3, 4)
    ax2.title.set_text('rec 0.5')
    ax2.imshow(np.transpose((rec_image_all[0][0]).detach().cpu().numpy(),
                            (1, 2, 0))[:216, :])

    ax3 = fig.add_subplot(3, 3, 5)
    ax3.title.set_text('rec 0.75')
    ax3.imshow(np.transpose((rec_image_all[1][0]).detach().cpu().numpy(),
                            (1, 2, 0))[:216, :])

    ax4 = fig.add_subplot(3, 3, 6)
    ax4.title.set_text('rec 1')
    ax4.imshow(np.transpose((rec_image_all[2][0]).detach().cpu().numpy(),
                            (1, 2, 0))[:216, :])
    
    ax5 = fig.add_subplot(3, 3, 7)
    ax5.title.set_text('loss 0.5')
    loss0 = ax5.imshow(np.transpose(
        (loss_image_all[0][0]).detach().cpu().numpy(), (1, 2, 0))[:216, :])
    fig.colorbar(loss0, ax=ax5)

    ax6 = fig.add_subplot(3, 3, 8)
    ax6.title.set_text('loss 0.75')
    loss1 = ax6.imshow(
        np.transpose((loss_image_all[1][0]).detach().cpu().numpy(),
                     (1, 2, 0))[:216, :])
    fig.colorbar(loss1, ax=ax6)
    
    ax7 = fig.add_subplot(3, 3, 9)
    ax7.title.set_text('loss 1')
    loss3 = ax7.imshow(
        np.transpose((loss_image_all[2][0]).detach().cpu().numpy(),
                     (1, 2, 0))[:216, :])
    fig.colorbar(loss3, ax=ax7)

    fig.savefig('polycraft_nov_det/eval/evaluation/problematic'
                + str(idx) + '.png')   # save the figure to file
    plt.close(fig)    # close the figure window

    return


def loss_vector_evaluation(model_paths, allts, plot_problematics=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bc_path = 'models/polycraft/binary_classification/threshold_selection_conv_v3_500.pt'
    classifier = ms_classifier.MultiscaleClassifierConvFeatComp(1, 429)
    classifier = model_utils.load_model(bc_path, classifier, device).eval()
    classifier.to(device)

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

    with torch.no_grad():
        for i, samples in enumerate(test_loader):

            loss_arrays = ()

            for n, model in enumerate(all_models):

                patches = samples[n][0][0]
                patches = torch.flatten(patches, start_dim=0, end_dim=1)
                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE
                ipt_shape = ipt_shapes[n]

                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)

                loss2d = rec_loss2d(x_rec, x)
                loss2d = torch.mean(loss2d, (1, 2, 3))  # avgd. per patch
                # Reshape loss values from flattened to "squared shape"
                loss2d = loss2d.reshape(1, 1, ipt_shape[0], ipt_shape[1])
                # Interpolate smaller scales such that they match scale 1
                loss2d = functional.resize(loss2d, (13, 15))
                loss_arrays = loss_arrays + (loss2d,)

            label = samples[0][1]

            if label == 0 or label == 1:
                target = True
                pos += 1
            if label == 2:
                target = False
                neg += 1

            # Classifier was trained over array of losses
            pred = classifier(loss_arrays)

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

    if plot_problematics:
        plot_fp_fn_examples(test_loader, all_models, classifier, t_opt, device)       
    del base_dataset
    return con_mat


if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    allthreshs = np.round(np.linspace(0.3, 1, 21), 4)
    cm = loss_vector_evaluation(paths, allthreshs, plot_problematics=False)
    print(cm)
    eval_plot.plot_con_matrix(cm).savefig(("con_matrix_conv_v3.png"))
  