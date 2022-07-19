from datetime import datetime
from pathlib import Path
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_vgg_model


def extract_features(feature_extractor, loader):
    """Extract features from trained VGG16 classifier backbone.
    Args:
        feature extractor (torch.nn.Module): Trained vgg16 classifier.
        loader (torch.utils.data.DataLoader): Dataloader to extract features from.
    Returns:
        features_list (list): Features and their corresponding targets.
    """
    # we don't need the classification head
    feature_extractor = feature_extractor.backbone
    features_list = []
    target_list = []

    normalizer = MinMaxScaler()

    for j, (data, target) in enumerate(loader):
        # novel --> 1, not novel (class 0, 1, 2, 3, 4) --> 0
        target = [0 if (i == 0 or i == 1 or i == 2 or i == 3 or i == 4) else (1) for i in target]
        target_list.append(target)
        print(data.shape)
        features = feature_extractor(data).detach().numpy()
        features_list.append(features)

    features_list = np.concatenate(features_list)
    target_list = np.concatenate(target_list)
    features_list = normalizer.fit_transform(features_list)

    return features_list, target_list


class OneClassSVMDetector(NoveltyDetector):
    def __init__(self, svm_path, classifier_path, device="cpu"):
        super().__init__(device)
        self.svm = joblib.load(svm_path)
        classifier = load_vgg_model(classifier_path, device).to(device).eval()
        # classification head is not needed
        self.feature_extractor = classifier.backbone

    def novelty_score(self, data):

        data = data.to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(data).cpu().detach().numpy()
            outputs = self.svm.decision_function(features)

        return torch.Tensor(outputs)


def fit_ocsvm(feature_extractor, train_loader, valid_loader):
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
    session_path = Path('OneClassSVM') / Path(start_time)
    print('--------------')
    features_train, targets_train = extract_features(feature_extractor, train_loader)
    features_valid, targets_valid = extract_features(feature_extractor, valid_loader)

    nu_range = np.linspace(0.1, 0.9, 9)
    gamma_range = np.logspace(-5, 1, 9)

    precision = np.zeros((len(nu_range), len(gamma_range)))
    recall = np.zeros((len(nu_range), len(gamma_range)))

    acc_nn = np.zeros((len(nu_range), len(gamma_range)))
    acc_n = np.zeros((len(nu_range), len(gamma_range)))

    for i, nu in enumerate(nu_range):
        for j, gamma in enumerate(gamma_range):
            print('--------------------------', flush=True)
            print('Nu ', nu)
            print('Gamma ', gamma)

            # OneClassSVM based on Schölkopf et. al.
            svm = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)

            # fit svm on training data using feature vectors (without any novelties)
            oc_svm = svm.fit(features_train)
            svm_pred = oc_svm.predict(features_train)
            svm_pred = [0 if (i >= 0) else 1 for i in svm_pred]

            # calculate accuracy
            train_acc = np.sum(svm_pred == targets_train)/len(features_train)
            print('Train Acc.: ', train_acc)

            # apply svm (fitted on training data) to validation data (with novelties)
            svm_pred = oc_svm.predict(features_valid)
            svm_pred = [0 if (i >= 0) else 1 for i in svm_pred]
            valid_acc = np.sum(svm_pred == targets_valid)/len(features_valid)
            print('Valid Acc.: ', valid_acc)

            corr_per_class = []
            num_per_class = []

            svm_pred = np.asarray([bool(x) for x in svm_pred])
            targets_valid = np.asarray([bool(x) for x in targets_valid])

            t_pos = np.sum(np.logical_and(svm_pred, targets_valid))
            f_pos = np.sum(np.logical_and(svm_pred, ~targets_valid))
            f_neg = np.sum(np.logical_and(~svm_pred, targets_valid))

            for c in range(2):  # we have only non-novel and novel class
                match = targets_valid == c
                noi_per_class = match.sum()
                num_per_class.append(noi_per_class)

                match = match & (svm_pred == targets_valid)
                num_corrects_per_class = match.sum()
                corr_per_class.append(num_corrects_per_class)

            print('Valid Acc, class non-novel: ', corr_per_class[0]/num_per_class[0], flush=True)
            print('Valid Acc, class novel: ', corr_per_class[1]/num_per_class[1], flush=True)

            precision[i][j] = t_pos/(t_pos + f_pos)
            recall[i][j] = t_pos/(t_pos + f_neg)
            print('precicion :', precision[i][j])
            print('recall :', recall[i][j])

            acc_nn[i][j] = corr_per_class[0]/num_per_class[0]
            acc_n[i][j] = corr_per_class[1]/num_per_class[1]

            # construct paths
            model_dir = Path("models") / session_path
            model_fname = Path("nu_%f_gamm_%f.pkl" % (nu, gamma))
            # make directory and save model
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(oc_svm, model_dir / model_fname)

    gamma_grid, nu_grid = np.meshgrid(gamma_range, nu_range)

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 5), dpi=300)
    p = ax[0].pcolor(gamma_grid, nu_grid, precision, vmin=abs(precision).min(),
                     vmax=abs(precision).max())
    ax[0].set_xscale('log')
    ax[0].set_xlabel('gamma')
    ax[0].set_ylabel('nu')
    ax[0].title.set_text('Precision')
    fig.colorbar(p, ax=ax[0])

    p = ax[1].pcolor(gamma_grid, nu_grid, recall, vmin=abs(recall).min(), vmax=abs(recall).max())
    ax[1].set_xscale('log')
    ax[1].set_xlabel('gamma')
    ax[1].set_ylabel('nu')
    ax[1].title.set_text('Recall')
    fig.colorbar(p, ax=ax[1])

    fig.savefig('param_search_prec_rec.png')

    fig2, ax2 = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 5), dpi=300)
    p = ax2[0].pcolor(gamma_grid, nu_grid, acc_nn, vmin=abs(acc_nn).min(), vmax=abs(acc_nn).max())
    ax2[0].set_xscale('log')
    ax2[0].set_xlabel('gamma')
    ax2[0].set_ylabel('nu')
    ax2[0].title.set_text('Accuracy non-novel')
    fig2.colorbar(p, ax=ax2[0])

    p = ax2[1].pcolor(gamma_grid, nu_grid, acc_n, vmin=abs(acc_n).min(), vmax=abs(acc_n).max())
    ax2[1].set_xscale('log')
    ax2[1].set_xlabel('gamma')
    ax2[1].set_ylabel('nu')
    ax2[1].title.set_text('Accuracy novel')
    fig.colorbar(p, ax=ax2[1])

    fig2.savefig('param_search_acc.png')

    return oc_svm


if __name__ == '__main__':
    from polycraft_nov_data.dataloader import novelcraft_dataloader
    from polycraft_nov_data.image_transforms import VGGPreprocess

    from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save

    output_parent = Path("models/vgg/eval_ocsvm")
    classifier_path = Path("models/vgg/vgg_classifier_1000.pt")

    nus = np.linspace(0.1, 0.9, 9)
    gammas = np.logspace(-5, 1, 9)

    for nu in nus:
        for gamma in gammas:
            print('gamma: ', gamma, ' nu: ', nu, flush=True)
            output_folder = output_parent / Path(f"nu={nu:.6f}_gamm={gamma:.6f}")
            model_path = output_parent / Path("fits") / Path(f"nu_{nu:.6f}_gamm_{gamma:.6f}.pkl")

            save_scores(
                OneClassSVMDetector(model_path, classifier_path, device=torch.device("cuda:0")),
                output_folder,
                novelcraft_dataloader("valid", VGGPreprocess(), 32),
                novelcraft_dataloader("test", VGGPreprocess(), 32))
            eval_from_save(output_folder)
