from datetime import datetime
import pathlib
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from polycraft_nov_data.dataloader import polycraft_dataloaders

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_vgg_model


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
    print('Lets collect features')
    
    normalizer = MinMaxScaler()

    for j, (data, target) in enumerate(loader):
        # novel --> 1, not novel (class 0, 1, 2, 3, 4) --> 0
        target = [0 if (i == 0 or i == 1 or i ==2 or i == 3 or i == 4) else (1) for i in target]
        target_list.append(target)
        print(data.shape)
        features = feature_extractor(data).detach().numpy()
        features_list.append(features)

        #if j == 1:
        #    break

    print('Features collected')
    features_list = np.concatenate(features_list)
    target_list = np.concatenate(target_list)
    print(features_list.shape)
    features_list = normalizer.fit_transform(features_list)
    
    return features_list, target_list


class OneClassSVMDetector(NoveltyDetector):
    def __init__(self, svm, device="cpu"):
        super().__init__(device)
        self.svm = svm
        
    def novelty_score(self, features):
        # svm outputs are positive for inliers, negative for outliers
        outputs = self.svm.decision_function(features)
        outputs = torch.Tensor(outputs)
        
        outputs = [0 if (i >= 0) else 1 for i in outputs]
        return outputs


def fit_oneclassSVM(feature_extractor, train_loader, valid_loader):
    """Fit a One-Class SVM. The SVM is fit based on features extracted by the
    trained VGG16 classifier backbone on the training set.
    Args:
        feature extractor (torch.nn.Module): Trained vgg16 classifier.
        train_loader (torch.utils.data.DataLoader): Training set for SVM.
        valid_loader (torch.utils.data.DataLoader): Validation set for SVM.
    Returns:
        svm (sklearn.svm.OneClassSVM): Fitted One-Class SVM model.
    """
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = pathlib.Path('OneClassSVM') / pathlib.Path(start_time)
    print('--------------')
    features_train, targets_train = extract_features(feature_extractor, train_loader)
    features_valid, targets_valid = extract_features(feature_extractor, valid_loader)

    nu_range = np.linspace(0.1, 0.9, 9)
    gamma_range = np.logspace(-5, 5, 11)
    
    precision = np.zeros((len(nu_range), len(gamma_range)))
    recall = np.zeros((len(nu_range), len(gamma_range)))

    for i, nu in enumerate(nu_range):
        for j, gamma in enumerate(gamma_range):
            print('--------------------------', flush = True)
            print('Nu ', nu)
            print('Gamma ', gamma)
            
            # OneClassSVM based on Schölkopf et. al.
            svm = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)

            # fit svm on training data using feature vectors (without any novelties)
            oc_svm = svm.fit(features_train)
            detector = OneClassSVMDetector(oc_svm)
            svm_pred = detector.novelty_score(features_train) 
            
            # calculate accuracy
            train_acc = np.sum(svm_pred == targets_train)/len(features_train)
            print('Train Acc.: ', train_acc)

            # apply svm (fitted on training data) to validation data (with novelties)
            svm_pred = detector.novelty_score(features_valid) 
            valid_acc = np.sum(svm_pred == targets_valid)/len(features_valid)
            print('Valid Acc.: ', valid_acc)

            corr_per_class = []
            num_per_class = []
            
            svm_pred = np.asarray([bool(x) for x in svm_pred])
            targets_valid = np.asarray([bool(x) for x in targets_valid])
            

            t_pos = np.sum(np.logical_and(svm_pred, targets_valid))
            f_pos = np.sum(np.logical_and(svm_pred, ~targets_valid))
            t_neg = np.sum(np.logical_and(~svm_pred, ~targets_valid))
            f_neg = np.sum(np.logical_and(~svm_pred, targets_valid))
            
            print('t_pos', t_pos)
            print('t_neg', t_neg)
            print('f_pos', f_pos)
            print('f_neg', f_neg)
            
            for c in range(2):  # we have only non-novel and novel class
                match = targets_valid == c
                noi_per_class = match.sum()
                num_per_class.append(noi_per_class)
                
                match = match & (svm_pred == targets_valid)
                num_corrects_per_class = match.sum()
                corr_per_class.append(num_corrects_per_class)

            print('tp', corr_per_class[1])
            print('tn', corr_per_class[0])

            print('Valid Acc, class non-novel: ', corr_per_class[0]/num_per_class[0], flush=True)
            print('Valid Acc, class novel: ', corr_per_class[1]/num_per_class[0], flush=True)

            precision[i][j] = t_pos/(t_pos + f_pos)
            recall[i][j] = t_pos/(t_pos + f_neg)

            # construct paths
            model_dir = pathlib.Path("models") / session_path
            model_fname = pathlib.Path("nu_%f_gamm_%f.pkl" % (nu, gamma))
            # make directory and save model
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(oc_svm, model_dir / model_fname)

    X, Y = np.meshgrid(gamma_range, nu_range)

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12,5), dpi=300)
    p = ax[0].pcolor(X, Y, precision, vmin=abs(precision).min(), vmax=abs(precision).max())
    ax[0].set_xscale('log')
    ax[0].set_xlabel('gamma')
    ax[0].set_ylabel('nu')
    ax[0].title.set_text('Precision')
    cb = fig.colorbar(p, ax=ax[0])

    p = ax[1].pcolor(X, Y, recall, vmin=abs(recall).min(), vmax=abs(recall).max())
    ax[1].set_xscale('log')
    ax[1].set_xlabel('gamma')
    ax[1].set_ylabel('nu')
    ax[1].title.set_text('Recall')
    cb = fig.colorbar(p, ax=ax[1])

    fig.savefig('param_search.png')

    return oc_svm


def test_oneclassSVM(feature_extractor, test_loader):
    """Test a trained One-Class SVM. The SVM is tested based on features extracted by the
    trained VGG16 classifier backbone on the test set.
    Args:
        feature extractor (torch.nn.Module): Trained vgg16 classifier.
        test_loader (torch.utils.data.DataLoader): Test set for SVM.
    """
    features_test, targets_test = extract_features(feature_extractor, test_loader)
    oc_svm = joblib.load('models/nu_0.300000_gamm_0.000100.pkl')
    detector = OneClassSVMDetector(oc_svm)
    svm_pred = detector.novelty_score(features_test)

    svm_pred = np.asarray([bool(x) for x in svm_pred])
    targets_test = np.asarray([bool(x) for x in targets_test])
    t_pos = np.sum(np.logical_and(svm_pred, targets_test))
    f_pos = np.sum(np.logical_and(svm_pred, ~targets_test))
    f_neg = np.sum(np.logical_and(~svm_pred, targets_test))

    precision = t_pos / (t_pos + f_pos)
    recall = t_pos / (t_pos + f_neg)

    print('Precision on test set: ', precision)
    print('Recall on test set: ', recall)

    return


if __name__ == '__main__':

    gpu = 1
    num_classes = 5
    device = torch.device(gpu if gpu is not None else "cpu")

    (train_loader, valid_loader, test_loader), labels = polycraft_dataloaders(batch_size=100,
                                                          image_scale=1.0,
                                                          patch=False,
                                                          include_novel=True,
                                                          shuffle=True, 
                                                          ret_class_to_idx=True)
    
    
    classifier = load_vgg_model(Path("../models/VGGPretrained_class_normal_fence_item_anvil_item_sand_item_coal_block/2022.05.10.11.51.34/1000.pt"), device)
    svm = test_oneclassSVM(classifier.eval(), test_loader)
    