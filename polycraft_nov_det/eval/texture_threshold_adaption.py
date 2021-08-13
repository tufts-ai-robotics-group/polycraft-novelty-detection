import torch
import itertools 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils import data
from skimage.feature import greycomatrix, greycoprops
import cv2

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst 
import polycraft_nov_data.image_transforms as image_transforms
import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
import polycraft_nov_data.dataset_transforms as dataset_transforms
import polycraft_nov_det.eval.plot as eval_plot

T_HISTOGRAM = 10

class SimpleAutoencoder(nn.Module):
    
    def __init__(self, in_feat, lat_dim, useBias=False):
        super(SimpleAutoencoder, self).__init__()     
        
        self.encoder = torch.nn.Linear(in_features=in_feat, out_features=lat_dim, bias=useBias)  
        self.decoder = torch.nn.Linear(in_features=lat_dim, out_features=in_feat, bias=useBias)
        
        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax( dim = 1)
        
    def forward(self, x):
        
        z = self.leakyrelu( self.encoder(x) )
        x_rec = self.relu( self.decoder(z) )
       
        return x_rec
        
def compute_tp_tn_fp_fn(valid_loader, model, model_hist, allts, device, pooling, 
                        scale):
    """
    Compute true positives, true negatives, false positives and false negatives
    (positive --> novel, negative --> non-novel)
    """
    
    loss_func2d = nn.MSELoss(reduction='none')
    
    FP = np.zeros(len(allts))
    TP = np.zeros(len(allts))
    FN = np.zeros(len(allts))
    TN = np.zeros(len(allts))
    t_hist = T_HISTOGRAM
    
    P, N = 0, 0
    
    with torch.no_grad():

        for i, sample in enumerate(valid_loader):
           
            patches = sample[0]
            x = patches.float().to(device)
            x.requires_grad = False
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))
            zeros = torch. zeros_like(loss2d)
            ones = torch.ones_like(loss2d)
            
            label = sample[1]  
                
            if label == 0 or label == 1:
                label = True
                P += 1
            if label == 2:
                label = False
                N += 1
            
            for ii, t in enumerate(allts):
                novelty_score = False
                thresholds = ones*t
                loss2d_bin = torch.where((loss2d > thresholds), 1, 0)
                
                if torch.sum(loss2d_bin) > 0:
                    novelty_score = True
                    
                if novelty_score is True and label is True:
                    TP[ii] += 1
                    
                if novelty_score is False and label is True:
                    FN[ii] += 1
                    
                if novelty_score is True and label is False:
                    FP[ii] += 1
                    
                if novelty_score is False and label is False:
                    TN[ii] += 1
                  
    return  P, N, FP, TP, FN, TN


def find_optimal_treshold(P, N, FP, TP, FN, TN, allts):
    """
    Compute optimal treshold based on true positives, true negatives, false 
    positives and false negatives using cost function. The cost of false
    positives and false negatives are set to 1 (for now).
    """
    frac_FP, frac_TP, frac_TN, frac_FN = FP/N, TP/P, TN/N, FN/P
    cost_FP, cost_TP, cost_TN, cost_FN = 1, 0, 0, 2
    total_cost = cost_FP * frac_FP + cost_TP * frac_TP + cost_TN * frac_TN + cost_FN * frac_FN
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = allts[optimal_idx]
   
    return optimal_threshold


def compute_confusion_matrix(t_opt, test_loader, model, model_hist, device, 
                             pooling):
    """
    Compute confusion matrix for the second half of the available unseen data 
    by using the optimal threshold we determined for the first half of the 
    unseen data. The optimal threshold is adapted according to the "wildness"
    of the texture, the lower the homogeneity of the ("seen") texture 
    the higher the threshold. The determination if a texture was
    seen during training or not is determined based on the histogram of the 
    patch. 
    """
    loss_func2d = nn.MSELoss(reduction='none')
    loss_func = nn.MSELoss()
    
    pred = []
    labels = []  # label 0 --> height, label 1 --> item, label 2 --> no novelty
    t_hist = T_HISTOGRAM
    
    with torch.no_grad():
    
        for i, sample in enumerate(test_loader):

            label = sample[1]  
            
            if label == 0 or label == 1:
                label = True
            if label == 2:
                label = False
                
            labels.append(label)
            
            patches = sample[0]
            x = patches.float().to(device)
            x.requires_grad = False

            histos = []
            homogeneities = []
            
            # Determine Homogeneity of patches
            for p in range(x.shape[0]):
                x_p = np.transpose(patches[p].detach().numpy(), (1, 2, 0))
                x_com = cv2.normalize(src=x_p, dst=None, alpha=0, beta=255, 
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                x_com  = cv2.cvtColor(x_com, cv2.COLOR_RGB2GRAY)
                
                glcm = greycomatrix(x_com, distances=[5], angles=[0], 
                                    levels=256, symmetric=True, normed=True)
                hom = greycoprops(glcm, 'homogeneity')[0, 0]
                
                histogram, bin_edges = np.histogram(
                    x_com[:, :], bins=256, range=(0, 256))
                histos.append(histogram)
                homogeneities.append(hom)
                
            histos = torch.FloatTensor(histos) 
            homogeneities = torch.FloatTensor(homogeneities).to(device)
            histos = histos.to(device)
            r_histos = model_hist(histos) 
            histos_loss = loss_func2d(histos, r_histos)
            histos_loss = torch.mean(histos_loss, 1)
            
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))
            zeros = torch. zeros_like(loss2d)
            ones = torch.ones_like(loss2d)
            
            thresholds = ones*t_opt
            max_thresh = ones*1.1  # maximum threshold
            # Increase t if texture was seen during training and it it's "wild"
            thresholds = torch.where((histos_loss > t_hist) , thresholds, 
                                     thresholds*(torch.minimum(max_thresh, 
                                     2 - homogeneities)))
            loss2d_bin = torch.where((loss2d > thresholds), 1, 0)
            
            novelty_score = False
               
            if torch.sum(loss2d_bin) > 0:
                novelty_score = True
            
            pred.append(novelty_score)

    cm = confusion_matrix(labels, pred)  # Determine confusion matrix

    return  cm
    

def texture_dependent_performance_evaluation():
    """
    Compute an "optimal" threshold, based on tp, fp, tn, fn on the validation 
    set, compute confusion matrix on test set. For the test set, we adjust the
    threshold selected for the validation set based on the "wildness" of the
    texture.
    """
    
    model_directory = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    model_path = 'models/polycraft/histogram/histogram_075_nz50_4000.pt'
    scale = 0.75
    n_z = 100
    pool = 'max'
    
    b_size = 1
    pc_input_shape = data_const.PATCH_SHAPE  # color channels, height, width
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    _, valid_loader, test_loader = polycraft_dataloaders(batch_size=1,
                                                            image_scale=scale, 
                                                            include_novel=True, 
                                                            shuffle=False)
    # construct model
    model = LSACIFAR10NoEst(pc_input_shape, n_z)
    model.load_state_dict(torch.load(model_directory, 
                                     map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    
    model_hist = SimpleAutoencoder(256, 16)
    model_hist.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_hist.eval()
    model_hist.to(device)
    
    if scale == 0.75 or scale == 0.5:
    
        #  max, 0.75, 0.5
        allts = np.round(np.linspace(0.003, 0.013, 30), 4) 
        allts = np.append(0.000005, allts)
        allts = np.append(0.0005, allts)
        allts = np.append(allts, 0.015)
        allts = np.append(allts, 0.018)
        allts = np.append(allts, 0.02)
        allts = np.append(allts, 0.03)
        allts = np.append(allts, 0.04)
        allts = np.append(allts, 0.05)
        allts = np.append(allts, 0.06)
        
    if scale == 1: 
    
        #  max, 1
        allts1 = np.round(np.linspace(0.003, 0.04, 40), 4) 
        allts2 = np.round(np.linspace(0.04, 0.07, 20), 4)
        allts = np.append(allts1, allts2)
    
    p, n, f_pos, t_pos, f_neg, t_neg = compute_tp_tn_fp_fn(valid_loader, 
                                               model, model_hist, allts, device, pool, 
                                               scale)
    
    thresh = find_optimal_treshold(p, n, f_pos, t_pos, f_neg, t_neg, allts)
    cm = compute_confusion_matrix(thresh, test_loader, model, model_hist,
                                  device, pool)
    
    eval_plot.plot_con_matrix(cm).savefig(("con_matrix_ms.png"))
    
if __name__ == '__main__':
    
    texture_dependent_performance_evaluation()
    
