from datetime import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.image_transforms import GaussianNoise

import polycraft_nov_det.plot as plot


def model_label(model):
    """Generate a label for the type of model being trained

    Args:
        model (torch.nn.Module): Model to generate label for.

    Returns:
        str: Label for the model.
    """
    model_label = type(model).__name__
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


def train_autoencoder(model, model_label, train_loader, valid_loader, lr, epochs=500,
                      train_noisy=True, gpu=None):
    """Train an autoencoder model.

    Args:
        model (torch.nn.Module): Model to train.
        model_label (str): Label for model type, preferably from model_label function.
        train_loader (torch.utils.data.DataLoader): Training set for model.
        valid_loader (torch.utils.data.DataLoader): Validation set for model.
        lr (float): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 500.
        train_noisy (bool, optional): Whether to use denoising autoencoder. Defaults to True.
        gpu (int, optional): Index of GPU to use, CPU if None. Defaults to None.

    Returns:
        torch.nn.Module: Trained autoencoder model.
    """
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = pathlib.Path(model_label) / pathlib.Path(start_time)
    # get Tensorboard writer
    writer = SummaryWriter(pathlib.Path("runs") / session_path)
    # define training constants
    loss_func = nn.MSELoss()
    device = torch.device(gpu if gpu is not None else "cpu")
    # move model to device
    model.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # train model
    for epoch in range(epochs):
        for mode in ["train", "valid"]:
            epoch_loss = 0
            epoch_count = 0
            # set per mode variables
            if mode == "train":
                loader = train_loader
                tag = "Average Train Loss"
            elif mode == "valid":
                loader = valid_loader
                tag = "Average Validation Loss"
            # iterate over data
            for data, target in loader:
                batch_size = data.shape[0]
                data = data.to(device)
                target = target.to(device)
                if mode == "train":
                    optimizer.zero_grad()
                # get model outputs and update weights with optimizer if training
                if not train_noisy or mode == "valid":
                    r_data, embedding = model(data)
                else:
                    r_data, embedding = model(GaussianNoise()(data))
                batch_loss = loss_func(data, r_data)
                if mode == "train":
                    batch_loss.backward()
                    optimizer.step()
                # logging
                epoch_loss += batch_loss.item() * batch_size
                epoch_count += batch_size
            # calculate and record train loss
            av_epoch_loss = epoch_loss / epoch_count
            writer.add_scalar(tag, av_epoch_loss, epoch)
        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            # get reconstruction visualization
            writer.add_figure("Reconstruction Vis", plot.plot_reconstruction(data, r_data), epoch)
            # save model
            save_model(model, session_path, epoch)
    return model


def train_classifier(model, model_label, encoder, train_loader, valid_loader, lr, epochs=500,
                     train_noisy=True, gpu=None):
    """Train an classifier model.

    Args:
        model (torch.nn.Module): Model to train.
        model_label (str): Label for model type, preferably from model_label function.
        encoder (torch.nn.Module): Encoder to produce inputs to model.
        train_loader (torch.utils.data.DataLoader): Training set for model.
        valid_loader (torch.utils.data.DataLoader): Validation set for model.
        lr (float): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 500.
        train_noisy (bool, optional): Whether to use denoising autoencoder. Defaults to True.
        gpu (int, optional): Index of GPU to use, CPU if None. Defaults to None.

    Returns:
        torch.nn.Module: Trained classifier model.
    """
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = pathlib.Path(model_label) / pathlib.Path(start_time)
    # get Tensorboard writer
    writer = SummaryWriter(pathlib.Path("runs") / session_path)
    # define training constants
    loss_func = nn.NLLLoss()
    device = torch.device(gpu if gpu is not None else "cpu")
    # move models to device
    model.to(device)
    encoder.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # disable gradient for encoder
    for param in encoder.parameters():
        param.requires_grad = False
    # train model
    for epoch in range(epochs):
        for mode in ["train", "valid"]:
            epoch_loss = 0
            epoch_count = 0
            # set per mode variables
            if mode == "train":
                loader = train_loader
                tag = "Average Train Loss"
            elif mode == "valid":
                loader = valid_loader
                tag = "Average Validation Loss"
            # iterate over data
            for data, target in loader:
                batch_size = data.shape[0]
                data = data.to(device)
                target = target.to(device)
                if mode == "train":
                    optimizer.zero_grad()
                # get model outputs and update weights with optimizer if training
                if not train_noisy or mode == "valid":
                    embedding = encoder(data)
                else:
                    embedding = encoder(GaussianNoise()(data))
                pred = model(embedding)
                batch_loss = loss_func(pred, target)
                if mode == "train":
                    batch_loss.backward()
                    optimizer.step()
                # logging
                epoch_loss += batch_loss.item() * batch_size
                epoch_count += batch_size
            # calculate and record train loss
            av_epoch_loss = epoch_loss / epoch_count
            writer.add_scalar(tag, av_epoch_loss, epoch)
        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            # save model
            save_model(model, session_path, epoch)
    return model
