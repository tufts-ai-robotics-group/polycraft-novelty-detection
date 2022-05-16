from datetime import datetime
import pathlib
import numpy as np
import joblib
import torch

from sklearn.svm import OneClassSVM

from polycraft_nov_data.dataloader import polycraft_dataloaders
import polycraft_nov_det.model_utils as model_utils


def extract_features(feature_extractor, loader):
    """Extract features from trained VGG16 classifier backbone.
    Args:
        feature extractor (torch.nn.Module): Trained vgg16 classifier.
        loader (torch.utils.data.DataLoader): Dataloader to extract features from.
    Returns:
        features_list, target_list (list, list): Features and their corresponding targets.
    """

    # we don't need the classification head
    feature_extractor = feature_extractor.backbone
    features_list = []
    target_list = []

    for j, (data, target) in enumerate(loader):
        # novel --> 1, not novel (class 0, 1, 2, 3, 4) --> 0
        target = [0 if (i == 0 or i == 1 or i ==2 or i == 3 or i == 4) else (1) for i in target]
        target_list.append(target)
        features = feature_extractor(data).detach().numpy()
        features_list.append(features)

        if j == 2:
            break

    print('Features collected')
    features_list = np.concatenate(features_list)
    target_list = np.concatenate(target_list)

    return features_list, target_list


def fit_oneclassSVM(feature_extractor, train_loader, valid_loader):
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
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = pathlib.Path('OneClassSVM') / pathlib.Path(start_time)
    features_train, targets_train = extract_features(feature_extractor, train_loader)
    features_valid, targets_valid = extract_features(feature_extractor, valid_loader)

    nu_range = np.linspace(0, 1, 6)
    gamma_range = np.logspace(-5, 5, 11)

    for nu in nu_range:
        for gamma in gamma_range:

            # OneClassSVM based on Schölkopf et. al.
            svm = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)

            # fit svm on training data using feature vectors (without any novelties)
            oc_svm = svm.fit(features_train)
            svm_pred = oc_svm.predict(features_train)
            svm_pred = [0 if (i == +1) else 1 for i in svm_pred]

            # calculate accuracy
            train_acc = np.sum(svm_pred == targets_train)/len(features_train)
            print('Train Acc.: ', train_acc)

            # apply svm (fitted on training data) to validation data (with novelties)
            svm_pred = oc_svm.predict(features_valid)
            svm_pred = [0 if (i == +1) else 1 for i in svm_pred]
            # calculate accuracy
            valid_acc = np.sum(svm_pred == targets_valid)/len(features_valid)
            print('Valid Acc.: ', valid_acc)

            acc_per_class = []

            for c in range(2):  # we have only non-novel and novel class
                match = targets_valid == c
                noi_per_class = match.sum()
                match = match & (svm_pred == targets_valid)
                num_corrects_per_class = match.sum()
                acc_per_class.append(num_corrects_per_class / noi_per_class)

            print('Valid Acc, class non-normal: ', acc_per_class[0], flush=True)
            print('Valid Acc, class novel: ', acc_per_class[1], flush=True)

            # construct paths
            model_dir = pathlib.Path("models") / session_path
            model_fname = pathlib.Path("nu_%d_gamm_%d.pkl" % (nu, gamma))
            # make directory and save model
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(oc_svm, model_dir / model_fname)

    return oc_svm


if __name__ == '__main__':

    gpu = 2
    num_classes = 5
    device = torch.device(gpu if gpu is not None else "cpu")

    (train_loader, valid_loader, _), labels = polycraft_dataloaders(batch_size=100,
                                                          image_scale=1.0,
                                                          patch=False,
                                                          include_novel=True,
                                                          shuffle=True, 
                                                          ret_class_to_idx=True)
    
    cl_path = '../models/VGGPretrained_class_normal_fence_item_anvil_item_sand_item_coal_block/2022.05.10.11.51.34/1000.pt'
    cl_path = pathlib.Path(cl_path)
    classifier = model_utils.load_polycraft_classifier(cl_path, device, num_classes=num_classes).eval()
    svm = fit_oneclassSVM(classifier, train_loader, valid_loader)
    