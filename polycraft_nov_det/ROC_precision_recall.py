import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from skimage import transform
import PIL
import torchvision
from sklearn import metrics

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet

class noramlizeImage(object):
    
    def __call__(self, image):
    
        minval = image.min()
        maxval = image.max()
        
        diff = maxval - minval
        
        if diff > 0:
            img_norm = (image - minval) / diff
        else:
            img_norm = torch.zeros(image.size())
        
        return img_norm

class cropImageChangeResolution(object):
    #Decrease resolution by scale_factor
   
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, image):
        
        image = np.asarray(image)
        image = image[0:234, :, :]
        image = transform.rescale(image, (self.scale_factor, self.scale_factor, 1), anti_aliasing = True)
        image = PIL.Image.fromarray(np.uint8(image*255))
       
        return image
    
class extractPatches(object):
    
    def __call__(self, image):
        
        p_size = 32
        #Extract patches
        image = np.asarray(image)
        stride = int(p_size/2) # patch stride
        image = torch.from_numpy(image)
        patches = image.unfold(0, p_size, stride).unfold(1, p_size, stride)
        
        return patches

def get_data_loader(scale, batch_size, data_dir):
    """
    Preprocess Images (remove minecraft bar, change image resolution,
    extract patches of size batch_size x batch_size in ordered way with batch_size/2 overlap
    and normalize between 0 and 1.
    
    :scale: scale used for decrease in resolution, set to 1 for no original resolution
    :param batch_size: size of the extracted batch
    :param data_dir: root directory where the dataset is
     
    :return: data laoder
    """
   
    trnsfrm = transforms.Compose([
            cropImageChangeResolution(scale),
            extractPatches(),
            #transforms.ToTensor(),
            noramlizeImage(),
            ])

    data = torchvision.datasets.ImageFolder(root = data_dir, transform = trnsfrm)
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=False)
    
    return loader

def compute_novelty_score_of_image(models_state_dict_path, allts, scale):
    
    b_size = 1
    pc_input_shape = (3, 32, 32)  # color channels, height, width
    n_z = 110
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    normal_loader = get_data_loader(scale, b_size, data_dir = 'datasets/normal_data/no_novelties')
    novel_loader = get_data_loader(scale, b_size, data_dir='datasets/novel_data/item_novelty')
    
    N = len(normal_loader) #all images are non-novel
    P = len(novel_loader) #all images are novel
    
    # construct model
    model = LSANet(pc_input_shape, n_z)
    model.load_state_dict(torch.load(models_state_dict_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
   
    loss_func2d = nn.MSELoss(reduction='none')
   
    FP = np.zeros(len(allts))
    TP = np.zeros(len(allts))
    
    FN = np.zeros(len(allts))
    TN = np.zeros(len(allts))
    
    
    print('Iterate through non-novel images.')
    
   
    for i, sample in enumerate(normal_loader):
        #print(i)
        
        patches = sample[0]
        flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

        x = flat_patches[0].float().to(device)
        x_rec, z = model(x)

        loss2d = loss_func2d(x_rec, x)
        loss2d = torch.mean(loss2d, (1, 2, 3)) #averaged loss per patch
        
        
        for ii, t in enumerate(allts):
            
            loss2d_t = torch.where(loss2d > t, 1, 0)
        
            novelty_score = any(loss2d_t) #True if novelty detected
            
            if novelty_score == True:
                #print(t, 'False positive, oh oh!')
                FP[ii] += 1
                
            if novelty_score == False:
                #print(t, "True negative!")
                TN[ii] += 1
    
    print('Iterate through novel images.')
        
    for i, sample in enumerate(novel_loader):
        #print(i)
    
        patches = sample[0]
        flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

        x = flat_patches[0].float().to(device)
        x_rec, z = model(x)
        
        loss2d = loss_func2d(x_rec, x) 
        loss2d = torch.mean(loss2d, (1, 2, 3))  #averaged loss per patch
        
        for ii, t in enumerate(allts):
            
            loss2d_t = torch.where(loss2d > t, 1, 0)
        
            novelty_score = any(loss2d_t) #True if novelty detected
            
            if novelty_score == True:
                #print(t, 'True negative')
                TP[ii] += 1
                
            if novelty_score == False:
                #print(t, "False negative, oh oh!")
                FN[ii] += 1
        
      
    return P, N, FP, TP, FN, TN
        
       

        
           
if __name__ == '__main__':
    
    state_dict_path = '../models/polycraft/saved_statedict_random_patches/saved_statedict_polycraft_scale_0_75/LSA_polycraft_no_est_075_random_3000.pt'
    
    # range of thresholds for scale 0.5
    #scale = 0.5
    #allts = np.round(np.linspace(0.0008, 0.006, 27), 5) 
    #allts = np.append(allts, 0.04)
    
    # range of thresholds for scale 0.75
    scale = 0.75
    allts = np.round(np.linspace(0.0052, 0.014, 23), 5) 
    allts = np.append(0.004, allts)
    allts = np.append(allts, 0.02)
    allts = np.append(allts, 0.08)
    
    
    # range of thresholds for scale 1
    #scale = 1
    #allts = np.round(np.linspace(0.008, 0.035, 28), 5)
    #allts = np.append(allts, 0.06)
    
    
    p, n, fp, tp, fn, tn = compute_novelty_score_of_image(state_dict_path, allts, scale)
    
    # ROC
    fpr = fp/n
    tpr = tp/p
    auc = -1 * np.trapz(tpr, fpr)
    
    # precision recall
    notzero = np.ones_like(tp)*1e-7
    
    #If necessary, prevent division by zero 
    allp = np.where((tp+fp) == 0, 1, (tp+fp))
    alln = np.where((tp+fn) == 0, 1, (tp+fn))
    
    prec = tp/allp
    prec = np.where(allp == 1, 1, prec)
    recall = tp/alln
    #prec = tp/(tp + fp)
    #recall = tp(tp + fn)
    
    auc_pr = metrics.auc(recall, prec)
    
     
    ### ROC plot ##############################################################
    
    plt.plot(fpr, tpr, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    i = 0
    for t in  allts:
        plt.text(fpr[i] * (1 + 0.03), tpr[i] * (1 + 0.01) , round(t,4), fontsize=12)
        i = i + 1
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    plt.show()
    
    ### Precision Recall curve ################################################
    
    plt.plot(recall, prec, linestyle='--', marker='o', color='m', lw = 2, clip_on=False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    i = 0
    for t in  allts:
        plt.text(recall[i] * (1 + 0.03), prec[i] * (1 + 0.01) , round(t,4), fontsize=12)
        i = i + 1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(' Precision Recall curve, AUC = %.2f'%auc_pr)
    plt.show()
    
    
    
    
    
   