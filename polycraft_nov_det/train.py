from datetime import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from polycraft_nov_data.image_transforms import GaussianNoise
from polycraft_nov_data.dataloader import polycraft_dataloaders_full_image
import polycraft_nov_data.data_const as data_const

import polycraft_nov_det.plot as plot
import polycraft_nov_det.models.vgg as vgg16


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


def train(model, model_label, train_loader, valid_loader, lr, epochs=500, train_noisy=True,
          gpu=None):
    """Train a model.
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
        torch.nn.Module: Trained model.
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
        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            # get reconstruction visualization
            writer.add_figure("Reconstruction Vis", plot.plot_reconstruction(data, r_data), epoch)
            # save model
            save_model(model, session_path, epoch)
    return model


def train_VGG(model, train_loader, valid_loader, lr, epochs=500, train_noisy=True, gpu=None):
    """Train a VGG model.
    Args:
        model (torch.nn.Module): (Pretrained) VGG Classifiction Model to train.
        train_loader (torch.utils.data.DataLoader): Training set for model.
        valid_loader (torch.utils.data.DataLoader): Validation set for model.
        lr (float): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 500.
        train_noisy (bool, optional): Whether to use denoising autoencoder. Defaults to True.
        gpu (int, optional): Index of GPU to use, CPU if None. Defaults to None.
    Returns:
        torch.nn.Module: Trained VGG model.
    """
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    model_label_ = model_label(model, include_classes=data_const.NORMAL_CLASSES)
    session_path = pathlib.Path(model_label_) / pathlib.Path(start_time)
    # get Tensorboard writer
    writer = SummaryWriter(pathlib.Path("runs_VGG") / session_path)
    # define training constants
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(gpu if gpu is not None else "cpu")
    # move model to device
    model.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    
    normal_class_indices = torch.tensor([0, 1, 11, 36, 53]) 
    
    # train model
    for epoch in range(epochs):
        print('---------------------------------------------', flush=True)
        print('Epoch Nr.', epoch, flush=True)
        train_loss = 0
        
        for data, target in train_loader:
            batch_size = data.shape[0]
            target_ = torch.empty_like(target)
            
            for data_idx in range(batch_size):
                for nci_idx, nci in enumerate(normal_class_indices):
                
                    if target[data_idx] == nci:
                        target_[data_idx] = torch.tensor(nci_idx, dtype=torch.long)
                        
                #plt.imshow(np.transpose(data[data_idx].detach().cpu(), (1, 2, 0)))
                #plt.title(str(target_[data_idx]))
                #plt.savefig('sanity_check/' + str(data_idx) + '.png') 
                              
            data = data.to(device)
            
            target_ = target_.to(device)
            
            optimizer.zero_grad()
            
            # update weights with optimizer
            if not train_noisy:
                pred = model(data)
            else:
                pred = model(GaussianNoise()(data))
            
            batch_loss = loss_func(pred, target_)
            batch_loss.backward()
            optimizer.step()
            # logging
            train_loss += batch_loss.item() * batch_size
            
       
        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        print('Avg. Train loss ', av_train_loss, flush = True)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        
        # get validation loss
        valid_loss = 0
        for data, target in valid_loader:
            
            batch_size = 0
            batch_size_max = data.shape[0]
            target_ = []
            data_ = []
            
            for data_idx in range(batch_size_max):
                for nci_idx, nci in enumerate(normal_class_indices):
                
                    # use only seen novelties for classifier validation
                    if target[data_idx] == nci:
                        target_.append(torch.tensor(nci_idx, dtype=torch.long))
                        data_.append(data[data_idx])
                        batch_size += 1
                        
            #for data_idx in range(batch_size):           
            #    plt.imshow(np.transpose(data_[data_idx].detach().cpu(), (1, 2, 0)))
            #    plt.title(str(target_[data_idx]))
            #    plt.savefig('sanity_check/' + str(data_idx) + '.png')
                        
            data_ = torch.stack(data_)
            target_ = torch.stack(target_)
            
            data_ = data_.to(device)
            target_ = target_.to(device)
            pred = model(data_)
            batch_loss = loss_func(pred, target_)
            valid_loss += batch_loss.item() * batch_size
            
        av_valid_loss = valid_loss / len(valid_loader)
        print('Avg. Valid loss ', av_valid_loss, flush = True)
        writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)
        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            # save model
            save_model(model, session_path, epoch)
       
    return model


if __name__ == '__main__':
    
    
    train_loader, valid_loader, _ = polycraft_dataloaders_full_image(batch_size=32, 
                                                          image_scale=1, 
                                                          include_novel=True, 
                                                          shuffle=True)
    
    print('Loading is done', flush=True)
    
    classifier = vgg16.VGGPretrained(num_classes=5)
    
    train_VGG(classifier, train_loader, valid_loader, lr=1e-5, epochs=1000, 
              train_noisy=True, gpu=1)