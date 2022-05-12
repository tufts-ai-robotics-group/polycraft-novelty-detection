from datetime import datetime
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.svm import OneClassSVM

import polycraft_nov_data.data_const as data_const
from polycraft_nov_data.image_transforms import GaussianNoise
from polycraft_nov_data.dataloader import polycraft_dataloaders

import polycraft_nov_det.plot as plot
import polycraft_nov_det.models.vgg as vgg16
import polycraft_nov_det.model_utils as model_utils


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
    
    
def calc_per_class_acc(preds, labels, noc):
    """Determine per-class accuracy.
    Args:
        preds (torch.tensor): Predictions computed by the model.
        labels (torch.tensor): Ground truth labels.
        noc (int): Number of classes the model uses.
    """
    acc_per_class = []
    pred_class = torch.argmax(preds, dim=1)  # softmax does not change order

    for c in range(noc):
        match = labels == c
        noi_per_class = match.sum()
        match = match & (pred_class == labels)
        num_corrects_per_class = match.sum()
        acc_per_class.append(num_corrects_per_class / noi_per_class)
       
    return torch.stack(acc_per_class)


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


def train_vgg(model, train_loader, valid_loader, num_classes, lr, epochs=500, gpu=None):
    """Train a VGG model.
    Args:
        model (torch.nn.Module): (Pretrained) VGG Classifiction Model to train.
        train_loader (torch.utils.data.DataLoader): Training set for model.
        valid_loader (torch.utils.data.DataLoader): Validation set for model.
        lr (float): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 500.
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

    # train model
    for epoch in range(epochs):
        print('---------------------------------------------', flush=True)
        print('Epoch Nr.', epoch, flush=True)
        train_loss = 0
        pred_list = []
        target_list = []
        model.train()
        for data, target in train_loader:
            batch_size = data.shape[0]
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # update weights with optimizer
            pred = model(data)
            pred_list.append(pred)
            target_list.append(target)
            batch_loss = loss_func(pred, target)
            batch_loss.backward()
            optimizer.step()
            # logging
            train_loss += batch_loss.item() * batch_size
        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        print('Avg. Train loss ', av_train_loss, flush=True)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        # calculate per class accuracy
        pred_list = torch.cat(pred_list)
        target_list = torch.cat(target_list)
        
        acc_pc = calc_per_class_acc(pred_list, target_list, num_classes).to(device)

        for c in range(num_classes):
            print('Avg. Train Acc, class ',str(c), ': ', acc_pc[c].item(), flush=True)
            writer.add_scalar("Average Train Acc class" + str(c), acc_pc[c].item(), epoch)

        with torch.no_grad():
            # get validation loss
            valid_loss = 0
            pred_list = []
            target_list = []
            model.eval()
            for data, target in valid_loader:
                batch_size = data.shape[0]
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                # update weights with optimizer
                pred = model(data)
                pred_list.append(pred)
                target_list.append(target)

                #for data_idx in range(batch_size):           
                #    plt.imshow(np.transpose(data[data_idx].detach().cpu(), (1, 2, 0)))
                #    plt.title(str(target[data_idx]))
                #    plt.savefig('sanity_check_valid/' + str(data_idx) + '.png')
 
                batch_loss = loss_func(pred, target)
                # logging
                valid_loss += batch_loss.item() * batch_size

            av_valid_loss = valid_loss / len(valid_loader)
            print('Avg. Valid loss ', av_valid_loss, flush=True)
            writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)
            # calculate per class accuracy
            pred_list = torch.cat(pred_list)
            target_list = torch.cat(target_list)
            acc_pc = calc_per_class_acc(pred_list, target_list, num_classes).to(device)
            for c in range(num_classes):
                print('Avg. Valid Acc, class ',str(c), ': ', acc_pc[c].item(), flush=True)
                writer.add_scalar("Average Valid Acc class" + str(c), acc_pc[c].item(), epoch)
            # updates every 10% of training time
            if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
                # save model
                save_model(model, session_path, epoch)
    return model


def fit_oneclassSVM(svm, feature_extractor, train_loader, valid_loader):
    """Fit a One-Class SVM. The SVM is fit based on features extracted by the
    trained VGG16 classifier backbone on the training set.
    Args:
        svm (sklearn.svm.OneClassSVM): One-Class SVM model.
        feature extractor (torch.nn.Module): Trained vgg16 classifier.
        train_loader (torch.utils.data.DataLoader): Training set for SVM.
        valid_loader (torch.utils.data.DataLoader): Validation set for SVM.
    Returns:
        svm (sklearn.svm.OneClassSVM): Fitted One-Class SVM model.
    """
    # we don't need the classification head
    feature_extractor = feature_extractor.backbone
    features_list = []
    target_list = []
   
    for j, (data, target) in enumerate(train_loader):
        # novel --> 1, not novel (class 0, 1, 2, 3, 4) --> 0
        target = [0 if (i == 0 or i == 1 or i ==2 or i == 3 or i == 4) else (1) for i in target]
        target_list.append(target)
        train_features = feature_extractor(data).detach().numpy()
        features_list.append(train_features)
           
    print('Features collected')
    features_list = np.concatenate(features_list)
    target_list = np.concatenate(target_list)
    # fit svm on training data using feature vectors (without any novelties)
    oc_svm = svm.fit(features_list)
    svm_pred = oc_svm.predict(features_list)
    svm_pred = [1 if (i == -1) else (0) for i in svm_pred]
    
    # calculate accuracy
    train_acc = np.sum(svm_pred == target_list)/len(features_list)
    print('Train Acc.: ', train_acc)
    
    features_list = []
    target_list = []

    for j, (data, target) in enumerate(valid_loader):
        # novel --> 1, not novel (class 0, 1, 2, 3, 4) --> 0
        target = [0 if (i == 0 or i == 1 or i ==2 or i == 3 or i == 4) else (1) for i in target]
        target_list.append(target)
        train_features = feature_extractor(data).detach().numpy()
        features_list.append(train_features)
      
    features_list = np.concatenate(features_list)
    target_list = np.concatenate(target_list)
    # apply svm (fitted on training data) to validation data (with novelties)
    svm_pred = oc_svm.predict(features_list)
    svm_pred = [1 if (i == -1) else (0) for i in svm_pred]
    # calculate accuracy
    valid_acc = np.sum(svm_pred == target_list)/len(features_list)
    print('Valid Acc.: ', valid_acc)
    
    acc_per_class = []
    
    for c in range(2):
        match = target_list == c
        noi_per_class = match.sum()
        match = match & (svm_pred == target_list)
        num_corrects_per_class = match.sum()
        acc_per_class.append(num_corrects_per_class / noi_per_class)
        
    print('Avg. Valid Acc, class non-normal: ', acc_per_class[0], flush=True)
    print('Avg. Valid Acc, class novel: ', acc_per_class[1], flush=True)
    return oc_svm


if __name__ == '__main__':

    gpu = 0
    num_classes = 5
    device = torch.device(gpu if gpu is not None else "cpu")

    # use vgg16 as feature extractor
    classifier = vgg16.VGGPretrained(num_classes=5)

    (train_loader, valid_loader, _), labels = polycraft_dataloaders(batch_size=100,
                                                          image_scale=1.0,
                                                          patch=False,
                                                          include_novel=True,
                                                          shuffle=True, 
                                                          ret_class_to_idx=True)
    
    model_path = 'models/VGGPretrained_class_normal_fence_item_anvil_item_sand_item_coal_block/2022.05.10.11.51.34/1000.pt'
    # OneClassSVM based on Schölkopf et. al.
    one_class_svm = OneClassSVM(nu=0.001, kernel='rbf', gamma='auto')
    model_path = pathlib.Path(model_path)
    model = model_utils.load_polycraft_classifier(model_path, device, num_classes=num_classes).eval()
    fit_oneclassSVM(one_class_svm, classifier, train_loader, valid_loader)