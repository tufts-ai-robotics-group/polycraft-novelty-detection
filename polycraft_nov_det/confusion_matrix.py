import torch
import itertools
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
from ROC_precision_recall import get_data_loader


def plot_confusion_matrix(cm, classes, t, scale, pool, cmap='BuPu'):
    # adapted from https://deeplizard.com/learn/video/0LhiS6yu2qQ
    # --> CNN Confusion Matrix with PyTorch - Neural Network Programming

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix, scale = %s, t = %s,  pooling = %s' % (
        str(scale), str(t), pool))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('conf.png', dpi=1000, bbox_inches="tight")


def compute_novelty_score_predictions(model_path, scale, thresh, pooling):
    """
    Apply model on non-novel and novel images and compute their novelty
    score predictions based on thresholding the pooled (max or mean) patch
    reconstruction errors.
    In general, we define novel as True, non-novel as False.
    :param model_path: path where the parameters of the trained model are
    stored
    :param scale: image_scale used for decrease in resolution,
    set to 1 for original resolution
    :param thresh: threshold used for novelty score computation
    :param pooling: type of pooling used for novelty score computation,
    either 'mean' or 'max'
    :return: labels (ground truth) and predictions
    """

    b_size = 1
    pc_input_shape = (3, 32, 32)  # color channels, height, width
    n_z = 110
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # temporary solution
    data_directory_nono = 'unseen_data/item_novelty'
    data_directory_no = 'unseen_data/no_novelties'

    normal_loader = get_data_loader(scale, b_size,
                                    data_dir=data_directory_nono)
    novel_loader = get_data_loader(scale, b_size, data_dir=data_directory_no)

    n = len(normal_loader)  # all images are non-novel
    p = len(novel_loader)   # all images are novel

    # construct model
    model = LSANet(pc_input_shape, n_z)
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    loss_func2d = nn.MSELoss(reduction='none')

    label_nono = np.zeros(n, dtype=bool)
    label_no = np.ones(p, dtype=bool)
    labels = np.append(label_nono, label_no)
    pred = []

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

            novelty_score = False

            if pooled_loss > thresh:
                novelty_score = True

            pred.append(novelty_score)

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

            novelty_score = False

            if pooled_loss > thresh:
                novelty_score = True

            pred.append(novelty_score)

    return labels, pred


if __name__ == '__main__':

    state_dict_path = '../models/polycraft/noisy/scale_0_75/1000.pt'

    pool = 'max'  # either maximum or mean of all patch losses is computed
    scale = 0.75
    t = 0.0076  # Threshold used to decide if something is novel or not
    classes = (['not novel', 'novel'])

    y_, y = compute_novelty_score_predictions(state_dict_path, scale, t, pool)
    y_ = torch.Tensor(y_)  # Ground truth, novel --> True, not novel --> False
    y = torch.Tensor(y)  # Prediction, novel --> True, not novel --> False

    cm = confusion_matrix(y_, y)
    plot_confusion_matrix(cm, classes, t, scale, pool)
