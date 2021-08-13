from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops

from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
import polycraft_nov_det.model_utils as model_utils


def eval_texture(model_path, device="cpu"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    model = model_utils.load_polycraft_model(model_path, device).eval()
    model = model.to(device)
    
    train_loader, valid_loader, test_loader = polycraft_dataloaders(
        batch_size=1, image_scale=0.75, include_novel=True, shuffle=False)
   
    # get targets determined at runtime
    base_dataset = polycraft_dataset()                                    
    loss_func = nn.MSELoss()
    
    all_homs = []
    all_loss = []
    
    with torch.no_grad():
        # Let's have a look at texture homogeneity of non-novel images.
        for i, sample in enumerate(train_loader):  
            if i == 50:
                break
            patch = sample[0]
           
            for p in range(patch.shape[0]):
                x_rgb = np.transpose(patch[p].detach().numpy(), (1, 2, 0))
                x_rgb = cv2.normalize(src=x_rgb, dst=None, alpha=0, beta=255, 
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # Conversion to grayscale to compute co-occurance matrix
                x_gray  = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2GRAY)
                com = greycomatrix(x_gray, distances=[5], angles=[0], 
                                   levels=256, symmetric=True, normed=True)
                # Compute homogeneity based on co-occurance matrix, defined in:
                # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html#skimage.feature.texture.greycoprops
                hom = greycoprops(com, 'homogeneity')[0, 0]
                all_homs.append(hom)
            
                x = patch[p].unsqueeze(0)
                x_rec, z = model(x.float().to(device))
               
                #MSE loss, averaged over patch
                loss = loss_func(x.to(device), x_rec.to(device))
                all_loss.append(loss.item())
                
    plt.scatter(all_homs, all_loss, c='#ff7f0e')
    plt.xlabel('Homogeneity')
    plt.ylabel('Rec. Loss')
    plt.savefig('hom_loss.png')        
  
    return 

"""          
if __name__ == '__main__':
    
    path = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    eval_texture(path)
"""    