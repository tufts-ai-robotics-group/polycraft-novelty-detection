from datetime import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.image_transforms import GaussianNoise

import polycraft_nov_det.plot as plot


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
        classes = "_".join([str(include_class) for include_class in include_classes])
        model_label += "class_" + classes
    return model_label


def save_model(model, model_label, epoch):
    """Save a model.

    Args:
        model (torch.nn.Module): Model to save.
        model_label (str): Label for the model type, preferably from model_label function.
        epoch (int): Training epoch to label saved model with.
    """
    # construct paths
    model_dir = pathlib.Path("models") / pathlib.Path(model_label)
    model_fname = pathlib.Path(model_label + "_%d.pt" % (epoch + 1,))
    # make directory and save model
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / model_fname)


def train(model, model_label, train_loader, valid_loader, lr, epochs=500, train_noisy=True):
    """Train a model.

    Args:
        model (torch.nn.Module): Model to train.
        model_label (str): Label for model type, preferably from model_label function.
        train_loader (torch.utils.data.DataLoader): Training set for model.
        valid_loader (torch.utils.data.DataLoader): Validation set for model.
        lr (float): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 500.
        train_noisy (bool, optional): Whether to use denoising autoencoder. Defaults to True.

    Returns:
        torch.nn.Module: Trained model.
    """
    # get Tensorboard writer
    writer = SummaryWriter("runs/" + model_label + "/" +
                           datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    # define training constants
    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # move model to device
    model.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # train model
    for epoch in range(epochs):
        train_loss = 0
        for data, target in train_loader:
            batch_size = data.shape[0]
            data = data.to(device)
            optimizer.zero_grad()
            # update weights with optimizer
            if not train_noisy:
                r_data, embedding = model(data)
            else:
                r_data, embedding = model(GaussianNoise()(data))
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            optimizer.step()
            # logging
            train_loss += batch_loss.item() * batch_size
        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        # get validation loss
        valid_loss = 0
        for data, target in valid_loader:
            batch_size = data.shape[0]
            data = data.to(device)
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            valid_loss += batch_loss.item() * batch_size
        av_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)
        # get reconstruction visualization
        writer.add_figure("Reconstruction Vis", plot.plot_reconstruction(data, r_data), epoch)
        # save model
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            save_model(model, model_label, epoch)
    return model
