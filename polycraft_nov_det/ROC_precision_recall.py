import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

from sklearn import metrics

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
import polycraft_nov_data.image_transforms as image_transforms


def get_data_loader(image_scale, batch_size, data_dir):
    """
    Preprocess Images (remove minecraft bar, change image resolution,
    extract patches of size batch_size x batch_size in ordered way with
    batch_size/2 overlap and normalize between 0 and 1.
    :param image_scale: image_scale used for decrease in resolution,
    set to 1 for original resolution
    :param batch_size: size of the extracted batch
    :param data_dir: root directory where the dataset is
    :return: data loader with batched images
    """
    trnsfrm = transforms.Compose([
        transforms.ToTensor(),
        image_transforms.CropUI(),
        image_transforms.ScaleImage(image_scale),
        image_transforms.ToPatches(),
    ])
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=trnsfrm)
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=False)
    return loader


def compute_novelty_scores_of_images(model_path, scale, allts, pooling):
    """
    Apply model on non-novel (--> negative) and novel ( --> positive) images
    and compute the amount of positives (P), negatives (N), false positives
    (FP), true positives (TP), false negatives (FN) and true negatives (TN)
    based on thresholding the pooled patch reconstruction errors.

    :param model_path: path where the parameters of the trained model are
    stored
    :param scale: image_scale used for decrease in resolution,
    set to 1 for original resolution
    :param allts: all threshold values used for novelty score computation
    :param pooling: type of pooling used for novelty score computation,
    either 'mean' or 'max'
    :return: P, N, FP, TP, FN, TN
    """
    b_size = 1
    pc_input_shape = (3, 32, 32)  # color channels, height, width
    n_z = 110
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # I store the files in the repository as a temporary solution, we
    # might add the BOX's URL to data_const!?
    data_directory_nono = 'unseen_data/item_novelty'
    data_directory_no = 'unseen_data/no_novelties'

    normal_loader = get_data_loader(scale, b_size, data_dir=data_directory_nono)
    novel_loader = get_data_loader(scale, b_size, data_dir=data_directory_no)

    n = len(normal_loader)  # all images are non-novel
    p = len(novel_loader)  # all images are novel

    # construct model
    model = LSANet(pc_input_shape, n_z)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    loss_func2d = nn.MSELoss(reduction='none')

    fp = np.zeros(len(allts))
    tp = np.zeros(len(allts))

    fn = np.zeros(len(allts))
    tn = np.zeros(len(allts))

    with torch.no_grad():

        print('Iterate through non-novel images.')
        for i, sample in enumerate(normal_loader):
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

                if novelty_score is True:
                    fp[ii] += 1

                if novelty_score is False:
                    tn[ii] += 1

        print('Iterate through novel images.')
        for i, sample in enumerate(novel_loader):
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

                if novelty_score is True:

                    tp[ii] += 1

                if novelty_score is False:

                    fn[ii] += 1

    return p, n, fp, tp, fn, tn


def plot_roc(p, n, fp, tp, allts):

    fpr = fp/n
    tpr = tp/p
    auc = -1 * np.trapz(tpr, fpr)

    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', marker='o', color='darkorange',
             lw=2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    i = 0
    for t in allts:
        plt.text(fpr[i] * (1 + 0.03), tpr[i] * (1 - 0.03), round(t, dec), fontsize=12)
        i = i + 1

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC = %.2f, scale = %s, pooling = %s' % (auc, str(scale), pool))
    plt.legend(loc="lower right")
    # plt.savefig('ROC.png')


def plot_precision_recall(tp, tn, fp, fn):

    # If necessary, prevent division by zero
    allp = np.where((tp+fp) == 0, 1, (tp+fp))
    alln = np.where((tp+fn) == 0, 1, (tp+fn))

    prec = tp/allp
    prec = np.where(allp == 1, 1, prec)
    recall = tp/alln

    auc_pr = metrics.auc(recall, prec)

    # fscore = (2 * prec * recall) / (prec + recall)
    # ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (allts[ix], fscore[ix]))

    plt.figure()
    plt.plot(recall, prec, linestyle='--', marker='o', color='m', lw=2, clip_on=False)
    plt.plot([0, 1], [0.5, 0.5], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    i = 0
    for t in allts:
        plt.text(recall[i] * (1 + 0.03), prec[i] * (1 - 0.03), round(t, dec), fontsize=12)
        i = i + 1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC = %.2f, scale = %s, pooling = %s' % (auc_pr, str(scale), pool))
    # plt.savefig('PR.png')


if __name__ == '__main__':

    state_dict_path = '../models/polycraft/saved_statedict_random_patches/saved_statedict_polycraft_scale_0_75/LSA_polycraft_no_est_075_random_3000.pt'
    pool = 'max'
    scale = 0.75
    dec = 4  # round thresholds to 4th decimal scale

    """
    # scale 0.5, mean
    allts = np.round(np.linspace(0.0006, 0.004, 30), dec)
    allts = np.append(allts, 0.008)

    # scale 0.5, max
    allts = np.round(np.linspace(0.002, 0.012, 30), dec)
    allts = np.append(0.001, allts)
    allts = np.append(allts, 0.04)

    # scale 0.75, mean
    allts = np.round(np.linspace(0.002, 0.004, 30), dec)
    allts = np.append(0.001, allts)
    allts = np.append(allts, 0.005)
    allts = np.append(allts, 0.008)

    """

    # scale 0.75, max
    allts = np.round(np.linspace(0.004, 0.025, 30), dec)
    allts = np.append(allts, 0.03)
    allts = np.append(allts, 0.04)
    allts = np.append(allts, 0.09)

    """

    # scale 1, mean
    allts = np.round(np.linspace(0.002, 0.008, 30), dec)

    # scale 1, max
    allts = np.round(np.linspace(0.01, 0.03, 30), dec)
    allts = np.append(0.005, allts)
    allts = np.append(allts, 0.05)
    allts = np.append(allts, 0.07)
    """

    p, n, fp, tp, fn, tn = compute_novelty_scores_of_images(state_dict_path, scale, allts, pool)

    plot_roc(p, n, fp, tp, allts)
    plot_precision_recall(tp, tn, fp, fn)
