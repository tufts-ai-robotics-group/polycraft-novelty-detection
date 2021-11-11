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
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst

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
    
    
def eval_polycraft_multiscale(model_paths, device="cpu", add_16x16_model=False):
    """
    Evaluate the binary classifier model trained on the three-dimensional rec.-error array 
    of each scale. The rec.-errors of the individual colour channels are not
    averaged for this model. At scale 0.5 and scale 0.75 we have one model each with
    patches of size 3x32x32, at scale 1 we have one model based on patch size
    3x32x32 and, optionally, an additional model based on patch size 3x16x16.
    Set add_16x16 to True if you want to use thi additonal fourth model.
    """

    # model_paths:
        # model_paths[0] --> binary classifier trained on rgb loss arrays
        # model_paths[1] --> scale 0.5 autoencoder model
        # model_paths[2] --> scale 0.75 autoencoder model
        # model_paths[3] --> scale 1 autoencoder model
        # model_paths[4] --> scale 1 autoencoder model (additional, optional)
        

    eval_path = Path(Path(model_paths[0]).parent, "eval_" + Path(model_paths[0]).stem)
    eval_path.mkdir(exist_ok=True)

    # Use the model trained on rec.-loss error
    bc_path = model_paths[0]
    classifier = model_utils.load_polycraft_classifier(bc_path, device=device, 
                                                       add_16x16_model=add_16x16_model)
    
    # Use 3x32x32 patch based model on all three scales (for now)
    model_path05 = Path(model_paths[1])
    model05 = model_utils.load_polycraft_model(model_path05, device).eval()
    model_path075 = Path(model_paths[2])
    model075 = model_utils.load_polycraft_model(model_path075, device).eval()
    model_path1 = Path(model_paths[3])
    model1 = model_utils.load_polycraft_model(model_path1, device).eval()
    
    all_models = [model05.to(device), model075.to(device), model1.to(device)]
    
    # If we use additional model trained on 3x16x16 patches at scale 1 
    if add_16x16_model:
        
        # we need this additonal model
        model_path1_16 = Path(model_paths[4])
        model1_16 = LSACIFAR10NoEst((3, 16, 16), 25)
        model1_16.load_state_dict(torch.load(model_path1_16, map_location=device))
        model1_16.eval()
        
        all_models.append(model1_16.to(device))
        
    detector = polycraft_nov_det.detector.ReconstructionDetMultiScale(
                                                    classifier, 
                                                    all_models, 
                                                    device=device, 
                                                    add_16x16_model=False)

    classifier.to(device)

    

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

    # Construct a dataset with the same images at each scale
    test_set = dataloader.TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # get targets determined at runtime
    base_dataset = polycraft_dataset()
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]

    # shapes of "patch array" for all scales.
    ipt_shapes = [[6, 7],  # Scale 0.5, patch size 3x32x32
                  [9, 11],  # Scale 0.75, patch size 3x32x32
                  [13, 15]]  # Scale 1, patch size 3x32x32

    # If we use additional model trained on 3x16x16 patches at scale 1 
    if add_16x16_model:

        # we need this additonal model
        model_path1_16 = Path(model_paths[4])
        model1_16 = LSACIFAR10NoEst((3, 16, 16), 25)
        model1_16.load_state_dict(torch.load(model_path1_16, map_location=device))
        model1_16.eval()

        # and patches of scale 1 of size 3x16x16
        _, _, test_set1_16 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=1, 
                                                            patch_shape=(3, 16, 16),
                                                            include_novel=True, 
                                                            shuffle=False)
        # Construct a dataset with the same images at each scale
        test_set = dataloader.QuattroDataset(test_set05, test_set075, 
                                             test_set1, test_set1_16)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        
        all_models = [model05.to(device), model075.to(device), 
                      model1.to(device), model1_16.to(device)]
        
        # shapes of "patch array" for all scales + one for the 16x16 model
        ipt_shapes = [[6, 7],  # Scale 0.5, patch size 3x32x32
                      [9, 11],  # Scale 0.75, patch size 3x32x32
                      [13, 15],  # Scale 1, patch size 3x32x32
                      [28, 31]]  # Scale 1, patch size 3x16x316
        
    pos, neg = 0, 0  # total number of novel (pos) and non-novel (neg) images
    thresholds = np.round(np.linspace(0.01, 1, 21), 4)
    alltps = np.zeros(len(thresholds))
    allfps = np.zeros(len(thresholds))
    alltns = np.zeros(len(thresholds))
    allfns = np.zeros(len(thresholds))

    with torch.no_grad():
        for i, samples in enumerate(test_loader):
            print(type(samples))
            pred = detector.is_novel(samples)
            label = samples[0][1]

            if label == 0 or label == 1:  # Item and height novelty
                target = True
                pos += 1
            if label == 2:  # No novolty
                target = False
                neg += 1

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
    
    return samples


if __name__ == '__main__':

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_classifier = 'models/polycraft/binary_classification/threshold_selection_conv_v3_rgb_500.pt'
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'

    paths = [path_classifier, path05, path075, path1]
    #sam = eval_polycraft_multiscale(paths, device=dev)

    path_classifier_16 =  'models/polycraft/binary_classification/threshold_selection_conv_v3_rgb_16_720.pt'
    path1_16 = 'models/polycraft/no_noise/scale_1_16/8000.pt'
    paths = [path_classifier_16, path05, path075, path1, path1_16]
    sam = eval_polycraft_multiscale(paths, device=dev, add_16x16_model=True)
    
  
   