import numpy as np

from polycraft_nov_det.data.cifar_loader import torch_cifar
import polycraft_nov_det.eval.stats as stats
from polycraft_nov_det.model_utils import load_disc_resnet


def eval_cifar10():
    # get dataloader
    norm_targets, novel_targets, (_, valid_loader, _) = torch_cifar(
        batch_size=128, include_novel=True)
    # get model instance
    model = load_disc_resnet("models/CIFAR10/discovery/200.pt",
                             len(norm_targets) + len(novel_targets), len(novel_targets))
    model.eval()
    # get model predictions
    device = "cpu"
    y_true = np.zeros((0,))
    y_pred = np.zeros((0,))
    for data, targets in valid_loader:
        data, targets = data.to(device), targets.to(device)
        label_pred, unlabel_pred, feat = model(data)
        unlabel_pred_max = np.argmax(unlabel_pred.detach().numpy(), axis=1)
        # select only unlabeled data
        norm_mask = targets < len(norm_targets)
        y_true = np.hstack((y_true, targets.numpy()[~norm_mask] - len(norm_targets)))
        y_pred = np.hstack((y_pred, unlabel_pred_max[~norm_mask]))
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    print(stats.cluster_acc(row_ind, col_ind, weight))
    print(stats.cluster_confusion(row_ind, col_ind, weight))
