from pathlib import Path

import numpy as np

from polycraft_nov_data import data_const as polycraft_const
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
from polycraft_nov_data.dataset_transforms import folder_name_to_target_list

import polycraft_nov_det.eval_calc as eval_calc
import polycraft_nov_det.eval_plot as eval_plot
import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.novelty as novelty


def eval_mnist(model_path, device="cpu"):
    model_path = Path(model_path)
    model = model_utils.load_mnist_model(model_path, device).eval()
    dataloaders = mnist_loader.torch_mnist(include_novel=True)
    eval(model_path, model, dataloaders, mnist_loader.MNIST_NORMAL,
         mnist_loader.MNIST_NOVEL, False, device)


def eval_polycraft(model_path, image_scale=1, device="cpu"):
    model_path = Path(model_path)
    model = model_utils.load_polycraft_model(model_path, device).eval()
    dataloaders = polycraft_dataloaders(image_scale=image_scale, include_novel=True)
    # get targets determined at runtime
    base_dataset = polycraft_dataset()
    normal_targets = folder_name_to_target_list(base_dataset,
                                                polycraft_const.NORMAL_CLASSES)
    novel_targets = folder_name_to_target_list(base_dataset,
                                               polycraft_const.NOVEL_CLASSES)
    del base_dataset
    eval(model_path, model, dataloaders, normal_targets, novel_targets, True, device)


def eval(model_path, model, dataloaders, normal_targets, novel_targets, pool_batches, device="cpu"):
    train_loader, valid_loader, test_loader = dataloaders
    eval_path = Path(model_path.parent, "eval_" + model_path.stem)
    eval_path.mkdir(exist_ok=True)
    # construct linear regularization
    train_lin_reg = model_utils.load_cached_lin_reg(model_path, model, train_loader)
    # fit detector threshold on validation set
    detector = novelty.ReconstructionDet(model, train_lin_reg, device)
    thresholds = np.linspace(0, 2, 50)
    t_pos, f_pos, t_neg, f_neg = eval_calc.confusion_stats(
        valid_loader, detector, thresholds, normal_targets, pool_batches)
    opt_index = eval_calc.optimal_index(t_pos, f_pos, t_neg, f_neg)
    opt_thresh = thresholds[opt_index]
    # plot PR and ROC curves on validation set
    eval_plot.plot_precision_recall(t_pos, f_pos, f_neg).savefig(eval_path / Path("pr.png"))
    eval_plot.plot_roc(t_pos, f_pos, t_neg, f_neg).savefig(eval_path / Path("roc.png"))
    # plot confusion matrix on test set
    t_pos, f_pos, t_neg, f_neg = eval_calc.confusion_stats(
        test_loader, detector, np.array([opt_thresh]), normal_targets, pool_batches)
    con_matrix = eval_calc.optimal_con_matrix(t_pos, f_pos, t_neg, f_neg)
    eval_plot.plot_con_matrix(con_matrix).savefig(eval_path / Path("con_matrix.png"))
