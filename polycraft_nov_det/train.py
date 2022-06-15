import collections
from datetime import datetime
import pathlib

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import polycraft_nov_det.eval.evals as evals
from polycraft_nov_det.loss import GCDLoss
from polycraft_nov_det.models.dino_train import DinoWithHead


def model_label(model, include_classes):
    """Generate a label for the type of model being trained

    Args:
        model (torch.nn.Module): Model to generate label for.
        include_classes (iterable): Classes model will be trained on.

    Returns:
        str: Label for the model.
    """
    model_label = type(model).__name__ + "_"
    if include_classes is None:
        model_label += "all_classes"
    else:
        classes = "_".join([str(int(include_class)) for include_class in include_classes])
        model_label += "class_" + classes
    return model_label


def save_model(model, session_path, epoch):
    """Save a model.

    Args:
        model (torch.nn.Module): Model to save.
        session_path (pathlib.Path): Unique path segment for the training session from train.
        epoch (int): Training epoch to label saved model with.
    """
    # construct paths
    model_dir = pathlib.Path("models") / session_path
    model_fname = pathlib.Path("%d.pt" % (epoch + 1,))
    # make directory and save model
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / model_fname)


def save_model_dino(model: DinoWithHead, session_path, epoch):
    """Save a fintuned DINO model. Separates head and last block weights.

    Args:
        model (DinoWithHead): Model to save.
        session_path (pathlib.Path): Unique path segment for the training session from train.
        epoch (int): Training epoch to label saved model with.
    """
    # construct paths
    model_dir = pathlib.Path("models") / session_path
    head_fname = pathlib.Path("head%d.pt" % (epoch + 1,))
    block_fname = pathlib.Path("block%d.pt" % (epoch + 1,))
    # extract state_dicts
    head_dict = model.head.state_dict()
    block_dict = collections.OrderedDict()
    for name, param in model.backbone.named_parameters():
        if "blocks.11" in name or "norm." in name:
            block_dict[name] = param
    # make directory and save model
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head_dict, model_dir / head_fname)
    torch.save(block_dict, model_dir / block_fname)


def run_epoch_gcd(labeled_loader, unlabeled_loader, model, loss_func: GCDLoss, device, epoch,
                  optimizer=None, lr_sched=None):
    is_train = optimizer is not None and lr_sched is not None
    if is_train:
        model.train()
    else:
        model.eval()
    loss = 0
    data_num = 0
    unlabeled_iter = iter(unlabeled_loader)
    for (data, t_data), targets in labeled_loader:
        # combine batch from labeled and unlabeled loaders
        try:
            (u_data, u_t_data), u_targets = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (u_data, u_t_data), u_targets = next(unlabeled_iter)
        data = torch.vstack([data, u_data])
        t_data = torch.vstack([t_data, u_t_data])
        targets = torch.hstack([targets, u_targets + loss_func.num_norm_targets])
        data, t_data, targets = data.to(device), t_data.to(device), targets.to(device)
        batch_size = data.shape[0]
        if is_train:
            optimizer.zero_grad()
        # calculate loss and backprop if training
        embeds, t_embeds = model(data), model(t_data)
        batch_loss = loss_func(embeds, t_embeds, targets)
        if is_train:
            batch_loss.backward()
            optimizer.step()
        # record loss without averaging
        loss += batch_loss.item() * batch_size
        data_num += batch_size
        # update lr scheduler
        if is_train:
            lr_sched.step()
    # calculate average loss
    av_loss = loss / data_num
    return av_loss


def train_gcd(model, model_label, labeled_loader, unlabeled_loader, norm_targets, lr=0.1,
              epochs=200, supervised_weight=0.35, gpu=None):
    """Train a model for generalized category discovery.

    Args:
        model (torch.nn.Module): Model to train.
        model_label (str): Label for model type, preferably from model_label function.
        labeled_loader (torch.utils.data.DataLoader): Labeled set for model.
        unlabeled_loader (torch.utils.data.DataLoader): Unlabeled set for model.
        norm_targets (list): Targets for normal data.
        lr (float): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 200.
        supervised_weight (float, optional): Weight for supervised GCD objective. Defaults to 0.35.
        gpu (int, optional): Index of GPU to use, CPU if None. Defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    """
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = pathlib.Path(model_label) / pathlib.Path(start_time)
    # get Tensorboard writer
    writer = SummaryWriter(pathlib.Path("runs") / session_path)
    # define training constants
    loss_func = GCDLoss(norm_targets, supervised_weight)
    device = torch.device(gpu if gpu is not None else "cpu")
    # move model to device
    model.to(device)
    # construct optimizer and lr scheduler (with per batch steps)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_batches = epochs * (len(labeled_loader) + len(unlabeled_loader))
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches, lr * .01)
    # train model
    for epoch in range(epochs):
        # calculate average train loss for epoch
        av_train_loss = run_epoch_gcd(
            labeled_loader, unlabeled_loader, model, loss_func, device, epoch, optimizer, lr_sched)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            # save model
            save_model_dino(model, session_path, epoch)
            # get validation accuracy
            valid_acc = evals.polycraft_gcd(model.backbone, device=device)
            writer.add_scalar("Average Validation Accuracy", valid_acc, epoch)
    return model
