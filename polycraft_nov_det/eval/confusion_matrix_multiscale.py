from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from polycraft_nov_data import data_const as polycraft_const
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
from polycraft_nov_data.dataset_transforms import folder_name_to_target_list
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.eval.plot as eval_plot


def eval_polycraft_multiscale(model_paths, device="cpu"):

    # threshods determined for (non-noisy) single scales with cost fp = cost fn = 1
    thresholds = [0.003, 0.0061, 0.0144]  # not noisy
    #thresholds = [0.003, 0.0058, 0.0144]  # noisy
    fac = 0.9  # Multiply thresholds by this factor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path05 = Path(model_paths[0])
    model05 = model_utils.load_polycraft_model(model_path05, device).eval()
    model_path075 = Path(model_paths[1])
    model075 = model_utils.load_polycraft_model(model_path075, device).eval()
    model_path1 = Path(model_paths[2])
    model1 = model_utils.load_polycraft_model(model_path1, device).eval()

    _, valid_loader05, test_loader05 = polycraft_dataloaders(batch_size=1,
                                                            image_scale=0.5, 
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_loader075, test_loader075 = polycraft_dataloaders(batch_size=1,
                                                            image_scale=0.75, 
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_loader1, test_loader1 = polycraft_dataloaders(batch_size=1,
                                                            image_scale=1, 
                                                            include_novel=True, 
                                                            shuffle=False)
    
    # get targets determined at runtime
    base_dataset = polycraft_dataset()                                         
    loss_func2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]
    
    pred = []
    labels = []  # label 0 --> height, label 1 --> items, label 2 --> no novelty
    
    # shapes of "patch array" for all scales.
    ipt_shapes = [[6, 7],
                  [9, 11],
                  [13, 15]]
    
    with torch.no_grad():

        print('Iterate through images.')
        for i, samples in enumerate(zip(valid_loader05, valid_loader075, valid_loader1)):
        
            binary_loss = []
            
            for n, model in enumerate(all_models):
                
                patches = samples[n][0]
                
                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE
                
                ipt_shape = ipt_shapes[n]
                
                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)
                
                loss2d = loss_func2d(x_rec, x)
                loss2d = torch.mean(loss2d, (1, 2, 3)) #averaged loss per patch
                # Reshape loss values of flattened patches to the "squared shape"
                loss2d = loss2d.reshape(ipt_shape[0], ipt_shape[1]).unsqueeze(0)
                # Interpolate smaller scales such that they match scale 1 
                loss2d = functional.resize(loss2d, (13, 15))
                ones = torch.ones_like(loss2d)
                zeros= torch.zeros_like(loss2d)
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
            pred.append(novelty_score)
            
    del base_dataset
    return labels, pred
  
"""          
if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    y_, y = eval_polycraft_multiscale(paths)
    cm = confusion_matrix(y_, y)
    eval_plot.plot_con_matrix(cm).savefig(("con_matrix_ms.png"))
"""  
   