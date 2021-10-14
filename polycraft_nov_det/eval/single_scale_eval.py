from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import functional
import matplotlib.pyplot as plt
import os
import itertools 
from sklearn import metrics

from polycraft_nov_data import data_const as polycraft_const
import polycraft_nov_det.model_utils as model_utils
from polycraft_nov_data.dataloader import polycraft_dataset_for_ms, polycraft_dataset
import polycraft_nov_det.eval.plot as eval_plot
import polycraft_nov_det.eval.binary_classification_training_positions as bctp 
import polycraft_nov_data.image_transforms as image_transforms
import polycraft_nov_det.models.multiscale_classifier as ms_classifier


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
    plt.savefig('ROC' + str(scale) + '.png')


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
    plt.title('AUC = %.2f, scale = %s, pooling = %s' % (auc_pr, str(scale), pool))
    plt.savefig('PR' + str(scale) + '.png')
    
    
def find_optimal_threshold(P, N, FP, TP, FN, TN, allts):
    """
    Compute optimal treshold based on true positives, true negatives, false
    positives and false negatives using cost function.
    """
    frac_FP, frac_TP, frac_TN, frac_FN = FP/N, TP/P, TN/N, FN/P
    cost_FP, cost_TP, cost_TN, cost_FN = 1, 0, 0, 1
    total_cost = (cost_FP * frac_FP + cost_TP * frac_TP + 
                  cost_TN * frac_TN + cost_FN * frac_FN)
    optimal_idx = np.argmin(total_cost)
    optimal_threshold = allts[optimal_idx]
    tp_opt = TP[optimal_idx]
    fn_opt = FN[optimal_idx]
    fp_opt = FP[optimal_idx]
    tn_opt = TN[optimal_idx]

    return tp_opt, fn_opt, fp_opt, tn_opt, optimal_threshold
    
    
def single_scale_evaluation(model_paths, allts_ms, plot_problematics=True):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
   

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
    valid_set = bctp.TrippleDataset(valid_set05, valid_set075, valid_set1)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    test_set = bctp.TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # get targets determined at runtime
    base_dataset = polycraft_dataset()
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]

    allscales = ['0.5', '0.75', '1']

    with torch.no_grad():
        
        for n, model in enumerate(all_models):
           
            allts = allts_ms[n]
            
            print('-----------------------------------------')
            print('Scale   ', allscales[n])
            
            alltps = np.zeros(len(allts))
            allfps = np.zeros(len(allts))
            alltns = np.zeros(len(allts))
            allfns = np.zeros(len(allts))
            tp, fp, tn, fn, pos, neg = 0, 0, 0, 0, 0, 0
            
            for i, samples in enumerate(valid_loader):
                label = samples[0][1]

                if label == 0 or label == 1:
                    target = True
                    pos += 1
                if label == 2:
                    target = False
                    neg += 1
                    
                patches = samples[n][0][0]
                patches = torch.flatten(patches, start_dim=0, end_dim=1)
                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE
                
                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)

                loss2d = rec_loss2d(x_rec, x)
                loss2d = torch.mean(loss2d, (1, 2, 3))  # avgd. per patch
                
                pooled_loss = torch.max(loss2d).item()  
                
                
                for ii, t in enumerate(allts):
                    if pooled_loss >= t and target is True:
                        alltps[ii] += 1
                    if pooled_loss >= t and target is False:
                        allfps[ii] += 1
                    if pooled_loss < t and target is False:
                        alltns[ii] += 1
                    if pooled_loss < t and target is True:
                        allfns[ii] += 1

            plot_roc(pos, neg, allfps, alltps, allscales[n], 'max')
            plot_precision_recall(alltps, alltns, allfps, allfns, allscales[n], 'max')
           

            tp, fn, fp, tn, t_opt = find_optimal_threshold(pos, neg, allfps, alltps,
                                                    allfns, alltns, allts)

            print('Optimal threshold', t_opt)

            con_mat = np.array([[tp, fn], [fp, tn]])
            
            eval_plot.plot_con_matrix(con_mat).savefig(("con_matrix_val" + allscales[n] + ".png"))
            
            tp, fp, tn, fn, pos, neg = 0, 0, 0, 0, 0, 0
            for i, samples in enumerate(test_loader):
                
                label = samples[0][1]

                if label == 0 or label == 1:
                    target = True
                    pos += 1
                if label == 2:
                    target = False
                    neg += 1
                    
                patches = samples[n][0][0]
                patches = torch.flatten(patches, start_dim=0, end_dim=1)
                _, ih, iw = polycraft_const.IMAGE_SHAPE
                _, ph, pw = polycraft_const.PATCH_SHAPE
                
                x = patches.float().to(device)
                x.requires_grad = False
                x_rec, z = model(x)

                loss2d = rec_loss2d(x_rec, x)
                loss2d = torch.mean(loss2d, (1, 2, 3))  # avgd. per patch
                
                pooled_loss = torch.max(loss2d).item()  
                
                if pooled_loss >= t_opt and target is True:
                    tp += 1
                if pooled_loss >= t_opt and target is False:
                    fp += 1
                if pooled_loss < t_opt and target is False:
                    tn += 1
                if pooled_loss < t_opt and target is True:
                    fn += 1

            con_mat = np.array([[tp, fn], [fp, tn]])
            
            eval_plot.plot_con_matrix(con_mat).savefig(("con_matrix_test" + allscales[n] + ".png"))

         
    del base_dataset
    return 


if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    
    allthreshs_ms = []
    
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
    
    allthreshs_ms.append(allts)  # thresholds for scale 0.5 
    allthreshs_ms.append(allts)  # and scale 0.75 
    
    allts1 = np.round(np.linspace(0.003, 0.04, 40), 4) 
    allts2 = np.round(np.linspace(0.04, 0.07, 20), 4)
    allts = np.append(allts1, allts2)
    
    allthreshs_ms.append(allts)  # thresholds for scale 1
    
    single_scale_evaluation(paths, allthreshs_ms, plot_problematics=True)
    
   
