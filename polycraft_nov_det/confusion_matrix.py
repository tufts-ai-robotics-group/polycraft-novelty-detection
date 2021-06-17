import torch
import torchvision
import itertools 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst 
import polycraft_nov_data.image_transforms as image_transforms
import polycraft_nov_data.data_const as data_const


# adapted from https://deeplizard.com/learn/video/0LhiS6yu2qQ 
# - CNN Confusion Matrix with PyTorch - Neural Network Programming
def plot_confusion_matrix(cm, classes, t, scale, pool, cmap='BuPu'):
    
    fig3 = plt.figure()
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


def get_data_loader(image_scale, batch_size, data_dir):
    """
    Preprocess Images (remove minecraft bar, change image resolution,
    extract patches of size batch_size x batch_size in ordered way with 
    batch_size/2 overlap normalized between 0 and 1. 
    :image_scale: scale used for decrease in resolution, 1 for original 
    resolution
    :param batch_size: size of the extracted patch
    :param data_dir: root directory where the dataset is
    :return: 2 dataloaders, one for ROC/Precision-Recall curve, one for 
    confusion matrix
    """
    trnsfrm = image_transforms.TestPreprocess(image_scale)
    
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=trnsfrm)
    total_noi = len(data)
    search_noi = int(0.5 * total_noi)  # Images used to find threshold, 50%
    test_noi = total_noi - search_noi  # Images used for confusion matrix, 50%

    search_dataset, test_dataset = torch.utils.data.random_split(data,
                                [search_noi, test_noi],
                                generator=torch.Generator().manual_seed(42))

    search_loader = torch.utils.data.DataLoader(search_dataset, batch_size, 
                                                shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, 
                                               shuffle=False)
    
    return search_loader, test_loader


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
    maxiter = np.minimum(N, P) 
    N = maxiter
    P = maxiter

    loss_func2d = nn.MSELoss(reduction='none')
    
    FP = np.zeros(len(allts))
    TP = np.zeros(len(allts))
    FN = np.zeros(len(allts))
    TN = np.zeros(len(allts))
    
    with torch.no_grad():

        print('Iterate through non-novel images.')
        for i, sample in enumerate(normaldata):
            if i == maxiter:
                break
            
            patches = sample[0]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)
    
            x = flat_patches[0].float().to(device)
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
                    
                if novelty_score == True:
                    FP[ii] += 1
                
                if novelty_score == False:
                    TN[ii] += 1

        print('Iterate through novel images.')
        for i, sample in enumerate(noveldata):

            if i == maxiter:
                break

            patches = sample[0]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)
    
            x = flat_patches[0].float().to(device)
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
                    
                if novelty_score == True:
                    TP[ii] += 1
                
                if novelty_score == False:
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
    print("Optimal threshold value is:", optimal_threshold)
    
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
    maxiter = np.minimum(N, P)
    N = maxiter
    P = maxiter
    
    label_nono = np.zeros((N), dtype=bool)  # Label all normal images as False
    label_no = np.ones((P), dtype=bool)  # Label all normal images as True
    labels = np.append(label_nono, label_no)
    pred = []
    
    with torch.no_grad():
    
        print('Iterate through non-novel images.')
        for i, sample in enumerate(normal_test):

            if i == maxiter:
                break

            patches = sample[0]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)
    
            x = flat_patches[0].float().to(device)
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

        print('Iterate through novel images.')
        for i, sample in enumerate(novel_test):

            if i == maxiter:
                break

            patches = sample[0]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)
    
            x = flat_patches[0].float().to(device)
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
    

def performance_evaluation(model_directory, scale, n_z):
    
    data_directory_nono = 'unseen_data/no_novelties'
    data_directory_no = 'unseen_data/item_novelty'

    pool = 'max'
    b_size = 1
    pc_input_shape = data_const.PATCH_SHAPE  # color channels, height, width
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
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
    
    normal_search, normal_test = get_data_loader(scale, b_size, 
                                                 data_dir=data_directory_nono)
    novel_search, novel_test = get_data_loader(scale, b_size, 
                                               data_dir=data_directory_no)
    
    p, n, fp, tp, fn, tn = compute_tp_tn_fp_fn(normal_search, novel_search, 
                                               model, allts, device, pool, 
                                               scale)
    
    thresh = find_optimal_treshold(p, n, fp, tp, fn, tn, allts)
    
    cm = compute_confusion_matrix(thresh, normal_test, novel_test, model,
                                  device, pool)
    classes = (['not novel', 'novel'])
    
    plot_confusion_matrix(cm, classes, thresh, scale, pool )
    

if __name__ == '__main__':
    model_dir = 'models/polycraft/noisy/scale_0_75/8000.pt'
    performance_evaluation(model_dir, 0.75, 100)
    