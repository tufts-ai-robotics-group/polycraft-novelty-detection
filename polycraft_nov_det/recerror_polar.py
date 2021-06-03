import torch
import torch.nn as nn
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst 
import polycraft_nov_data.image_transforms as image_transforms
import polycraft_nov_data.data_const as data_const


def plot_images_instead_of_dots(x, y, images, ax=None):
    ax = ax or plt.gca()
    idx = 0
     
    for xi, yi in zip(x, y):
        img = images[idx]  # get single patch
        img = np.transpose(img[0, :, :, :], (1, 2, 0))  
       
        im = OffsetImage(img, zoom=0.4)
        idx += 1
        im.image.axes = ax

        ab = AnnotationBbox(im, (xi, yi), frameon=False, pad=0.0,)
        ax.add_artist(ab)
        

def plot_error_with_patch_2d(values, patches):
    """
    The reconstruction error is plotted in a polar coordinate system, the angle
    of a rec. error plot point is set randomly, the distance from the 
    coordinate system's center point is the rec. error plot point is the
    reconstruction error itself. The rec. error plot point is then depicted
    as the patch it was computed of. 
    
    Args:
       all_losses (list of floats): List of the rec. errors of all patches
       all_patches (list of torch.tensors): List of all the input patches
            corresponding to each rec. error in all_losses
    """
    Phi = 2*np.pi*np.random.rand(len(values))  # compute random angle
    
    allx = []
    ally = []
   
    fig, ax = plt.subplots()

    for i in range(len(values)):
        r = values[i]
        phi = Phi[i]
        ax.plot(r*np.cos(phi), r*np.sin(phi))
        
        allx.append(r*np.cos(phi))
        ally.append(r*np.sin(phi))
    
    plot_images_instead_of_dots(allx, ally, patches, ax=ax)
      
    plt.grid(True)
    plt.xlim((- max(values), max(values)))
    plt.ylim((- max(values), max(values)))    
    plt.gca().set_aspect("equal")   
    plt.title('Rec errors in polar coord sys with random angle')

        
def get_data_loader(scale, data_dir):
    """Get images, convert them to tensors, remove Minecraft bar,
    rescale them and sample patches randomly. 
    
    Args:
        scale (float): Scaling to apply to image, 1.0 for original resolution
        data_dir (string): Folder where subfolders with images are saved in
        
    Returns:
        loader (torch.DataLoader): Dataloader with randomly sampled patches 
    """
    trnsfrm = image_transforms.TrainPreprocess(scale)
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=trnsfrm)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    
    return loader


def compute_novelty_scores_of_images(model_path, scale, noi):
    """
    Apply model on patches and compute the averaged reconstruction error over 
    each patch.
    
    Args:
        model_path (string):Path where the parameters of the trained model are 
            stored 
        scale (float): Scaling to apply to image, 1.0 for original resolution
        noi (int): Number of images/patches we want to use and plot 
    
    Returns:
        all_losses (list of floats): List of the rec. errors of all patches
        all_patches (list of torch.tensors): List of all the input patches
            corresponding to each rec. error in all_losses
    """
    n_z = 110
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    data_directory_nono = 'datasets/normal_data'
    loader = get_data_loader(scale, data_dir=data_directory_nono)
    
    # construct model
    model = LSACIFAR10NoEst(data_const.PATCH_SHAPE, n_z)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
   
    loss_func2d = nn.MSELoss(reduction='none')
    all_losses = []
    all_patches = []
    
    with torch.no_grad():
        
        ctr = 0
        print('Iterate through non-novel images.')
        for i, sample in enumerate(loader):
           
            x = sample[0].to(device)  # shape: B x C x H x W 
            x.requires_grad = False
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))  # averaged loss per patch
            
            all_losses.append(loss2d.item())
            all_patches.append(x) 
            
            ctr += 1
            
            if (ctr == noi):
                break
                
    return all_losses, all_patches

    
if __name__ == '__main__':
    
    plt.rcParams['figure.dpi'] = 800
    state_dict = '../models/polycraft/saved_statedict_random_patches/saved_statedict_polycraft_scale_0_75/LSA_polycraft_no_est_075_random_3000.pt'
    scale = 0.75
    
    losses, patches = compute_novelty_scores_of_images(state_dict, scale, 200)
    plot_error_with_patch_2d(losses, patches)
    
   
