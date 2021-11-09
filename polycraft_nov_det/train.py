from datetime import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import functional

from polycraft_nov_data.image_transforms import GaussianNoise
from polycraft_nov_data.data_const import PATCH_SHAPE, IMAGE_SHAPE

import polycraft_nov_det.models.multiscale_classifier as ms_classifier
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.plot as plot
import polycraft_nov_data.dataloader as dataloader
from polycraft_nov_data.dataloader import polycraft_dataset


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


def train_ensemble_classifier(model_paths, lr=0.0003, epochs=500, gpu=None, 
                              add_16x16_model=False):
    """Train a binary classifier on the "rec.-error arrays" computed by trained
    autoencoder models. We have 3-dimensional arrays since we compute the error
    between each colour channel. 
       
    Args:
        model_paths (list): List with the parameter dictionaries of the trained
        models:
        # model_paths[0] --> scale 0.5 autoencoder model (3x32x32)
        # model_paths[1] --> scale 0.75 autoencoder model (3x32x32)
        # model_paths[2] --> scale 1 autoencoder model (3x32x32)
        
        optional:
        # model_paths[3] --> scale 1 autoencoder model (3x16x16)    
            
        lr (float, optional): Learning rate.
        epochs (int, optional): Number of epochs to train for. Defaults to 500.
        gpu (int, optional): Index of GPU to use, CPU if None. Defaults to None.
        add_16x16_model (bool): If set to True, the additional 3x16x16 patch
        model at scale 1 is used as well.
    Returns:
        torch.nn.Module: Trained model.
    """
    device = torch.device(gpu if gpu is not None else "cpu")
    
     # If we use additional model trained on 3x16x16 patches at scale 1 as well
    if add_16x16_model:
        classifier = ms_classifier.MultiscaleClassifierConvFeatComp4Models(3, 572)
    
    # Otherwise just use models trained on 3x32x32 patches at each scale  
    else:
        classifier = ms_classifier.MultiscaleClassifierConvFeatComp3Models(3, 429)
    
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr)
    BCEloss = nn.BCELoss()
    
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = pathlib.Path('ensemble_classifier') / pathlib.Path(start_time)
    # get Tensorboard writer
    writer = SummaryWriter(pathlib.Path("runs") / session_path)
    
    model_path05 = pathlib.Path(model_paths[0])
    model05 = model_utils.load_polycraft_model(model_path05, device).eval()
    model_path075 = pathlib.Path(model_paths[1])
    model075 = model_utils.load_polycraft_model(model_path075, device).eval()
    model_path1 = pathlib.Path(model_paths[2])
    model1 = model_utils.load_polycraft_model(model_path1, device).eval()

    _, valid_set05, test_set05 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.5, 
                                                            patch_shape=PATCH_SHAPE,
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_set075, test_set075 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.75,
                                                            patch_shape=PATCH_SHAPE,
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_set1, test_set1 = dataloader.polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=1, 
                                                            patch_shape=PATCH_SHAPE,
                                                            include_novel=True, 
                                                            shuffle=False)

    valid_set = dataloader.TrippleDataset(valid_set05, valid_set075, valid_set1)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)

    test_set = dataloader.TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # get targets determined at runtime
    base_dataset = polycraft_dataset()
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]

    # shapes of "patch array" for all scales.
    ipt_shapes = [[6, 7],
                  [9, 11],
                  [13, 15]]

    # If we use additional model trained on 3x16x16 patches at scale 1 
    if add_16x16_model:

        # we need this additonal model
        model_path1_16 = pathlib.Path(model_paths[3])
        model1_16 = LSACIFAR10NoEst((3, 16, 16), 25)
        model1_16.load_state_dict(torch.load(model_path1_16, 
                                             map_location=device))
        model1_16.eval()

        # and patches of scale 1 of size 3x16x16
        _, valid_set1_16, test_set1_16 = dataloader.polycraft_dataset_for_ms(
                                                            batch_size=1,
                                                            image_scale=1, 
                                                            patch_shape=(3, 16, 16),
                                                            include_novel=True, 
                                                            shuffle=False)
        
        # Construct a dataset with the same images at each scale
        valid_set = dataloader.QuattroDataset(valid_set05, valid_set075,
                                             valid_set1, valid_set1_16)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
        
        # Construct a dataset with the same images at each scale
        test_set = dataloader.QuattroDataset(test_set05, test_set075,
                                             test_set1, test_set1_16)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        all_models = [model05.to(device), model075.to(device),
                      model1.to(device), model1_16.to(device)]

        # shapes of "patch array" for all scales + one for the 16x16 model
        ipt_shapes = [[6, 7],  # Scale 0.5, patch size 3x32x32
                      [9, 11],  # Scale 0.75, patch size 3x32x32
                      [13, 15],  # Scale 1, patch size 3x32x32
                      [28, 31]]  # Scale 1, patch size 3x16x316

    # train model
    print('Start Training', flush=True)
    for epoch in range(epochs):
        print('Epoch number  ', epoch, flush=True)
        train_loss = 0
        valid_loss = 0
        train_acc = 0
        valid_acc = 0

        for i, samples in enumerate(valid_loader):

            loss_arrays = ()

            with torch.no_grad():

                for n, model in enumerate(all_models):

                    patches = samples[n][0][0]
                    patches = torch.flatten(patches, start_dim=0, end_dim=1)
                    _, ih, iw = IMAGE_SHAPE
                    _, ph, pw = PATCH_SHAPE
                    ipt_shape = ipt_shapes[n]

                    x = patches.float().to(device)
                    x.requires_grad = False
                    x_rec, z = model(x)

                    loss2d = rec_loss2d(x_rec, x)  # #patches x 3 x 32 x 32
                    loss2d = torch.mean(loss2d, (2, 3))  # avgd. per patch
                    # Reshape loss values from flattened to "squared shape"
                    loss2d_r = loss2d[:, 0].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                    loss2d_g = loss2d[:, 1].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                    loss2d_b = loss2d[:, 2].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                    # Concatenate to a "3 channel" loss array
                    loss2d_rgb = torch.cat((loss2d_r, loss2d_g), 1)
                    loss2d_rgb = torch.cat((loss2d_rgb, loss2d_b), 1)
                    # Interpolate smaller scales such that they match scale 1
                    loss2d = functional.resize(loss2d_rgb, (13, 15))
                    loss_arrays = loss_arrays + (loss2d,)

            label = samples[0][1]

            if label == 0 or label == 1:
                target = torch.ones((1, 1)).to(device)
            if label == 2:
                target = torch.zeros((1, 1)).to(device)

            optimizer.zero_grad()
            pred = classifier(loss_arrays)
            loss = BCEloss(pred, target)
            loss.backward()
            optimizer.step()

            if pred >= 0.5 and target == torch.ones((1, 1)).to(device):
                train_acc += 1
            if pred < 0.5 and target == torch.zeros((1, 1)).to(device):
                train_acc += 1

            # logging
            train_loss += loss.item()

        train_loss = train_loss / (len(valid_loader))
        train_acc = train_acc / (len(valid_loader))
        writer.add_scalar("Average Train Loss", train_loss, epoch)
        print('Average training loss  ', train_loss, flush=True)
        writer.add_scalar("Average Train Acc", train_acc, epoch)
        print('Average training Acc  ', train_acc, flush=True)

        for i, samples in enumerate(test_loader):

            loss_arrays = ()

            with torch.no_grad():

                for n, model in enumerate(all_models):

                    patches = samples[n][0][0]
                    patches = torch.flatten(patches, start_dim=0, end_dim=1)
                    _, ih, iw = IMAGE_SHAPE
                    _, ph, pw = PATCH_SHAPE
                    ipt_shape = ipt_shapes[n]

                    x = patches.float().to(device)
                    x.requires_grad = False
                    x_rec, z = model(x)

                    loss2d = rec_loss2d(x_rec, x)  # #patches x 3 x 32 x 32
                    loss2d = torch.mean(loss2d, (2, 3))  # avgd. per patch
                    # Reshape loss values from flattened to "squared shape"
                    loss2d_r = loss2d[:, 0].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                    loss2d_g = loss2d[:, 1].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                    loss2d_b = loss2d[:, 2].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                    # Concatenate to a "3 channel" loss array
                    loss2d_rgb = torch.cat((loss2d_r, loss2d_g), 1)
                    loss2d_rgb = torch.cat((loss2d_rgb, loss2d_b), 1)
                    # Interpolate smaller scales such that they match scale 1
                    loss2d = functional.resize(loss2d_rgb, (13, 15))
                    loss_arrays = loss_arrays + (loss2d,)

            label = samples[0][1]

            if label == 0 or label == 1:
                target = torch.ones((1, 1)).to(device)
            if label == 2:
                target = torch.zeros((1, 1)).to(device)

            pred = classifier(loss_arrays)
            loss = BCEloss(pred, target)

            if pred >= 0.5 and target == torch.ones((1, 1)).to(device):
                valid_acc += 1
            if pred < 0.5 and target == torch.zeros((1, 1)).to(device):
                valid_acc += 1

            # logging
            valid_loss += loss.item()

        valid_loss = valid_loss / (len(test_loader))
        valid_acc = valid_acc / (len(test_loader))
        writer.add_scalar("Average Validation Loss", valid_loss, epoch)
        writer.add_scalar("Average Validation Acc", valid_acc, epoch)
        print('Average Validation loss  ', valid_loss, flush=True)
        print('Average Validation Acc  ', valid_acc, flush=True)

        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
            # save model
            save_model(classifier, session_path, epoch)

    del base_dataset
    return


if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    #train_ensemble_classifier(paths, lr=0.0003, epochs=500, gpu=1, 
    #                          add_16x16_model=False)
    
    path_classifier_16 =  'models/polycraft/binary_classification/threshold_selection_conv_v3_rgb_16_720.pt'
    path1_16 = 'models/polycraft/no_noise/scale_1_16/8000.pt'
    paths = [path05, path075, path1, path1_16]
    train_ensemble_classifier(paths, lr=0.0003, epochs=500, gpu=1, 
                              add_16x16_model=True)
    
   