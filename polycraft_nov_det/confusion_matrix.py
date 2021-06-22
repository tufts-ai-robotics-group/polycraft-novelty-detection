import torch
import itertools 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils import data

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst 
import polycraft_nov_data.image_transforms as image_transforms
import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
import polycraft_nov_data.dataset_transforms as dataset_transforms


# adapted from https://deeplizard.com/learn/video/0LhiS6yu2qQ 
# - CNN Confusion Matrix with PyTorch - Neural Network Programming
def plot_confusion_matrix(cm, classes, t, scale, pool, cmap='BuPu'):
    
    fig3 = plt.figure()
    print('Confusion matrix, optimal threshold is ', t)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix, scale = %s, t = %s,  pooling = %s' 
              % (str(scale), str(t), pool))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    color_thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > color_thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig3.savefig('conf.png', bbox_inches="tight")


def polycraft_dataloaders_50_50(batch_size=32, include_classes=None, 
                                image_scale=1.0, shuffle=True,
                                all_patches=False):
    """torch DataLoaders for Polycraft datasets

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        include_classes (list, optional): List of names of classes to include.
                                          Defaults to None, including all classes.
        image_scale (float, optional): Scaling applied to images. Defaults to 1.0.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        all_patches (bool, optional): Whether to replace batches with all patches from an image.
                                      Defaults to False.

    Returns:
        (DataLoader, DataLoader): Polycraft dataloader with novelty.
                                              Contains batches of (3, 32, 32) images,
                                              with values 0-1.
    """
    # if using patches, override batch dim to hold the set of patches
    if not all_patches:
        collate_fn = None
        transform = image_transforms.TrainPreprocess(image_scale)
    else:
        batch_size = None
        collate_fn = dataset_transforms.collate_patches
        transform = image_transforms.TestPreprocess(image_scale)
    # get the dataset
    dataset = polycraft_dataset(transform)
    # update include_classes to use indices instead of names
    include_classes = dataset_transforms.folder_name_to_target(dataset, include_classes)
    # split into datasets
    search_set, test_set = dataset_transforms.filter_split(
        dataset, [.5, .5], include_classes
    )
    # get DataLoaders for datasets
    num_workers = 4
    prefetch_factor = 1 if batch_size is None else max(batch_size//num_workers, 1)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
    }
    return (data.DataLoader(search_set, **dataloader_kwargs),
            data.DataLoader(test_set, **dataloader_kwargs))


def plot_roc(p, n, fp, tp, scale, pool):
    """
    Plot ROC curve based on previously determined false and true positives.
    """
    fpr = fp/n
    tpr = tp/p
    auc = -1 * np.trapz(tpr, fpr)

    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', marker='o', color='darkorange',
             lw=2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC = %.2f, scale = %s, pooling = %s' % (auc, str(scale), pool))
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')


def plot_precision_recall(tp, tn, fp, fn, scale, pool):
    """
    Plot Precision Recall curve based on previously determined false and true
    positives and false and true negatives.
    """
    # If necessary, prevent division by zero
    allp = np.where((tp+fp) == 0, 1, (tp+fp))
    alln = np.where((tp+fn) == 0, 1, (tp+fn))

    prec = tp/allp
    prec = np.where(allp == 1, 1, prec)
    recall = tp/alln

    auc_pr = metrics.auc(recall, prec)

    plt.figure()
    plt.plot(recall, prec, linestyle='--', marker='o', color='m', lw=2, 
                                                                 clip_on=False)
    plt.plot([0, 1], [0.5, 0.5], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC = %.2f, scale = %s, pooling = %s' % 
                                                    (auc_pr, str(scale), pool))
    plt.savefig('PR.png')
   
    
def compute_tp_tn_fp_fn(normaldata, noveldata, model, allts, device, pooling, 
                        scale):
    """
    Compute true positives, true negatives, false positives and false negatives
    (positive --> novel, negative --> non-novel)
    """
    N = len(normaldata)  # all images are non-novel
    P = len(noveldata)  # all images are novel
    print('Number of non-novel images used to determine threshold: ', N)
    print('Number of novel images used to determine threshold: ', P)
    
    loss_func2d = nn.MSELoss(reduction='none')
    
    FP = np.zeros(len(allts))
    TP = np.zeros(len(allts))
    FN = np.zeros(len(allts))
    TN = np.zeros(len(allts))
    
    with torch.no_grad():

        for i, sample in enumerate(normaldata):
           
            patches = sample[0]
            
            x = patches.float().to(device)
            x.requires_grad = False
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))  # averaged loss per patch
            
            if pooling == 'mean':
                # mean of all patch losses
                pooled_loss = torch.mean(loss2d).item()  
            if pooling == 'max':
                # maximum of all patch losses
                pooled_loss = torch.max(loss2d).item()  
            
            for ii, t in enumerate(allts):
                novelty_score = False
                
                if pooled_loss > t:
                    novelty_score = True
                    
                if novelty_score is True:
                    FP[ii] += 1
                
                if novelty_score is False:
                    TN[ii] += 1

        for i, sample in enumerate(noveldata):

            patches = sample[0]
            
            x = patches.float().to(device)
            x.requires_grad = False
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))  # averaged loss per patch
            
            if pooling == 'mean':
                pooled_loss = torch.mean(loss2d).item()  
            if pooling == 'max':
                pooled_loss = torch.max(loss2d).item()  
                
            for ii, t in enumerate(allts):
                novelty_score = False
                if pooled_loss > t:
                    novelty_score = True
                    
                if novelty_score is True:
                    TP[ii] += 1
                
                if novelty_score is False:
                    FN[ii] += 1  
                    
    plot_roc(P, N, FP, TP, scale, pooling)
    plot_precision_recall(TP, TN, FP, FN, scale, pooling)

    return  P, N, FP, TP, FN, TN


def find_optimal_treshold(P, N, FP, TP, FN, TN, allts):
    """
    Compute optimal treshold based on true positives, true negatives, false 
    positives and false negatives using cost function. The cost of false
    positives and false negatives are set to 1 (for now).
    """
    frac_FP, frac_TP, frac_TN, frac_FN = FP/N, TP/P, TN/N, FN/P
    cost_FP, cost_TP, cost_TN, cost_FN = 1, 0, 0, 1
    total_cost = cost_FP * frac_FP + cost_TP * frac_TP + cost_TN * frac_TN + cost_FN * frac_FN
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = allts[optimal_idx]
   
    return optimal_threshold

def compute_confusion_matrix(t_opt, normal_test, novel_test, model, device, 
                             pooling):
    """
    Compute confusion matrix for the second half of the available unseen data 
    by using the optimal threshold we determined for the first half of the 
    unseen data.
    """
    loss_func2d = nn.MSELoss(reduction='none')
    N = len(normal_test)  # all images are non-novel
    P = len(novel_test) # all images are novel
   
    print('Number of non-novel images used for confusion matrix: ', N)
    print('Number of novel images used for confusion matrix: ', P)
   
    label_nono = np.zeros((N), dtype=bool)  # Label all normal images as False
    label_no = np.ones((P), dtype=bool)  # Label all normal images as True
    labels = np.append(label_nono, label_no)
    pred = []
    
    with torch.no_grad():
    
        for i, sample in enumerate(normal_test):

            patches = sample[0]
            
            x = patches.float().to(device)
            x.requires_grad = False
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3)) # averaged loss per patch
            
            if pooling == 'mean':
                pooled_loss = torch.mean(loss2d).item()  
            if pooling == 'max':
                pooled_loss = torch.max(loss2d).item()  
            
            novelty_score = False
               
            if pooled_loss > t_opt:
                novelty_score = True
            
            pred.append(novelty_score)

        for i, sample in enumerate(novel_test):

            patches = sample[0]
            
            x = patches.float().to(device)
            x.requires_grad = False
            x_rec, z = model(x)
    
            loss2d = loss_func2d(x_rec, x)
            loss2d = torch.mean(loss2d, (1, 2, 3))  # averaged loss per patch
            
            if pooling == 'mean':
                pooled_loss = torch.mean(loss2d).item()
            if pooling == 'max':
                pooled_loss = torch.max(loss2d).item()
              
            novelty_score = False
             
            if pooled_loss > t_opt:
                novelty_score = True
            
            pred.append(novelty_score)
            
    cm = confusion_matrix(labels, pred)  # Determine confusion matrix

    return  cm
    

def performance_evaluation():
    """
    Computes ROC and precision-recall curve for one (unseen) subset of non-novel
    and novel images. Then these can be used to compute an "optimal" threshold,
    which we evaluate on the other (unseen) subset of non-novel and novel 
    images.
    """
    
    model_directory = './models/polycraft/noisy/scale_1/8000.pt'
    scale = 1
    n_z = 100
    pool = 'max'
    
    b_size = 1
    pc_input_shape = data_const.PATCH_SHAPE  # color channels, height, width
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Let's use valid. set for threshold selection, test set for conf. matrix
    _, normal_search, normal_test = polycraft_dataloaders(batch_size=1, 
                                    include_classes=['normal'], 
                                    image_scale=scale, all_patches=True)
    
    # Split novelty images in two sets and use one for threshold selection,
    # the other for conf. matrix
    novel_search, novel_test = polycraft_dataloaders_50_50(batch_size=1, 
                               include_classes=['item'], 
                               image_scale=scale, all_patches=True)
    
    # construct model
    model = LSACIFAR10NoEst(pc_input_shape, n_z)
    model.load_state_dict(torch.load(model_directory, 
                                     map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    
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
    
    p, n, fp, tp, fn, tn = compute_tp_tn_fp_fn(normal_search, novel_search, 
                                               model, allts, device, pool, 
                                               scale)
    
    thresh = find_optimal_treshold(p, n, fp, tp, fn, tn, allts)
    cm = compute_confusion_matrix(thresh, normal_test, novel_test, model,
                                  device, pool)
    
    classes = (['not novel', 'novel'])
    plot_confusion_matrix(cm, classes, thresh, scale, pool)
    

if __name__ == '__main__':
    
    performance_evaluation()
    
