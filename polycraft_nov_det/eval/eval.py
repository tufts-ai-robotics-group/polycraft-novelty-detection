from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional

from polycraft_nov_data import data_const as polycraft_const
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
from polycraft_nov_data.dataset_transforms import folder_name_to_target_list
from polycraft_nov_data.data_const import PATCH_SHAPE

import polycraft_nov_data.dataloader as dataloader
import polycraft_nov_det.eval.plot as eval_plot
import polycraft_nov_det.eval.stats as eval_stats
import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.models.multiscale_classifier as ms_classifier
import polycraft_nov_det.detector


def eval_mnist(model_path, device="cpu"):
    model_path = Path(model_path)
    model = model_utils.load_mnist_model(model_path, device).eval()
    dataloaders = mnist_loader.torch_mnist(include_novel=True)
    eval_base(model_path, model, dataloaders, mnist_loader.MNIST_NORMAL,
              mnist_loader.MNIST_NOVEL, False, device)


def eval_polycraft(model_path, image_scale=1, device="cpu"):
    model_path = Path(model_path)
    model = model_utils.load_polycraft_model(model_path, device).eval()
    dataloaders = polycraft_dataloaders(image_scale=image_scale, include_novel=True)
    # get targets determined at runtime
    base_dataset = polycraft_dataset()
    normal_targets = folder_name_to_target_list(base_dataset,
                                                polycraft_const.NORMAL_CLASSES)
    novel_targets = folder_name_to_target_list(base_dataset,
                                               polycraft_const.NOVEL_CLASSES)
    del base_dataset
    eval_base(model_path, model, dataloaders, normal_targets, novel_targets, True, device)


def eval_base(model_path, model, dataloaders, normal_targets, novel_targets, pool_batches,
              device="cpu"):
    train_loader, valid_loader, test_loader = dataloaders
    eval_path = Path(model_path.parent, "eval_" + model_path.stem)
    eval_path.mkdir(exist_ok=True)
    # construct linear regularization
    train_lin_reg = model_utils.load_cached_lin_reg(model_path, model, train_loader)
    # fit detector threshold on validation set
    detector = polycraft_nov_det.detector.ReconstructionDet(model, train_lin_reg, device)
    thresholds = np.linspace(0, 2, 200)
    t_pos, f_pos, t_neg, f_neg = eval_stats.confusion_stats(
        valid_loader, detector, thresholds, normal_targets, pool_batches)
    opt_index = eval_stats.optimal_index(t_pos, f_pos, t_neg, f_neg)
    opt_thresh = thresholds[opt_index]
    # plot PR and ROC curves on validation set
    eval_plot.plot_precision_recall(t_pos, f_pos, f_neg).savefig(eval_path / Path("pr.png"))
    eval_plot.plot_roc(t_pos, f_pos, t_neg, f_neg).savefig(eval_path / Path("roc.png"))
    # plot confusion matrix on test set
    t_pos, f_pos, t_neg, f_neg = eval_stats.confusion_stats(
        test_loader, detector, np.array([opt_thresh]), normal_targets, pool_batches)
    con_matrix = eval_stats.optimal_con_matrix(t_pos, f_pos, t_neg, f_neg)
    eval_plot.plot_con_matrix(con_matrix).savefig(eval_path / Path("con_matrix.png"))
    
    
def eval_polycraft_multiscale(model_paths, device="cpu"):
    
    # model_paths:
        # model_paths[0] --> scale 0.5 autoencoder model
        # model_paths[1] --> scale 0.75 autoencoder model
        # model_paths[2] --> scale 1 autoencoder model
        # model_paths[3] --> binary classifier trained on rgb loss arrays
        
    eval_path = Path(Path(model_paths[3]).parent, "eval_" + Path(model_paths[3]).stem)
    eval_path.mkdir(exist_ok=True)
    
    # Use the model trained on rec.-loss error   
    bc_path = model_paths[3]
    classifier = ms_classifier.MultiscaleClassifierConvFeatComp(3, 429)
    classifier = model_utils.load_model(bc_path, classifier, device).eval()
    classifier.to(device)
    
    # Use 3x32x32 patch based model on all three scales (for now)
    model_path05 = Path(model_paths[0])
    model05 = model_utils.load_polycraft_model(model_path05, device).eval()
    model_path075 = Path(model_paths[1])
    model075 = model_utils.load_polycraft_model(model_path075, device).eval()
    model_path1 = Path(model_paths[2])
    model1 = model_utils.load_polycraft_model(model_path1, device).eval()
    
    _, _, test_set05 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.5, 
                                                            patch_shape=PATCH_SHAPE,
                                                            include_novel=True, 
                                                            shuffle=False)
    _, _, test_set075 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.75,
                                                            patch_shape=PATCH_SHAPE,
                                                            include_novel=True, 
                                                            shuffle=False)
    _, _, test_set1 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=1, 
                                                            patch_shape=PATCH_SHAPE,
                                                            include_novel=True, 
                                                            shuffle=False)
    
    test_set = dataloader.TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # get targets determined at runtime
    base_dataset = polycraft_dataset()
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]

    pos, neg = 0, 0
    thresholds = np.round(np.linspace(0.01, 1, 21), 4)
    alltps = np.zeros(len(thresholds))
    allfps = np.zeros(len(thresholds))
    alltns = np.zeros(len(thresholds))
    allfns = np.zeros(len(thresholds))

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

                loss2d = rec_loss2d(x_rec, x)  # #patches x 3 x 32 x 32
                loss2d = torch.mean(loss2d, (2, 3))  # avgd. per patch
                # Reshape loss values from flattened to "squared shape"
                loss2d_r = loss2d[:, 0].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                loss2d_g = loss2d[:, 1].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                loss2d_b = loss2d[:, 2].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                # Concatenate to a "3 channel" loss array
                loss2d_rgb = torch.cat((loss2d_r, loss2d_g), 1)
                loss2d_rgb = torch.cat((loss2d_rgb, loss2d_b), 1)
                # Interpolate smaller scales such that they match scale 1
                loss2d = functional.resize(loss2d_rgb, (13, 15))
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

            for ii, t in enumerate(thresholds):

                if pred >= t and target is True:
                    alltps[ii] += 1
                if pred >= t and target is False:
                    allfps[ii] += 1
                if pred < t and target is False:
                    alltns[ii] += 1
                if pred < t and target is True:
                    allfns[ii] += 1
                    
    con_matrix = eval_stats.optimal_con_matrix(alltps, allfps, alltns, allfns)        
    opt_index = eval_stats.optimal_index(alltps, allfps, alltns, allfns)
    opt_thresh = thresholds[opt_index]
   
    eval_plot.plot_con_matrix(con_matrix).savefig(eval_path / Path("con_matrix.png"))
     
    del base_dataset
    

"""    
if __name__ == '__main__':
    
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    path_classifier = 'models/polycraft/binary_classification/threshold_selection_conv_v3_rgb_500.pt'
    paths = [path05, path075, path1, path_classifier]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_polycraft_multiscale(paths, device=dev)
"""    
   