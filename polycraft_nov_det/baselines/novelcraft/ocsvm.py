from datetime import datetime
from pathlib import Path
import numpy as np
import joblib
import torch

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_vgg_model


def extract_features(feature_extractor, loader, device):
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
    
    #max_features = 50000

    for j, (data, target) in enumerate(loader):
        
        # novel --> 1, not novel (class 0, 1, 2, 3, 4) --> 0
        target = [0 if (i == 0 or i == 1 or i == 2 or i == 3 or i == 4) else 
                  (1) for i in target]
        target_list.append(target)
        features = feature_extractor(data.to(device)).cpu().detach().numpy()
        features_list.append(features)
        
        #if j > max_features:
            #break
        
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


def fit_ocsvm(feature_extractor, train_loader, valid_loader, nu_range, gamma_range, device):
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
    features_train, targets_train = extract_features(feature_extractor, 
                                                     train_loader, device)
    
    
    print(features_train.shape)
    
    features_valid, targets_valid = extract_features(feature_extractor, 
                                                     valid_loader, device)

    for nu in nu_range:
        for gamma in gamma_range:
            print('--------------------------', flush=True)
            print('Nu ', nu)
            print('Gamma ', gamma)

            # OneClassSVM based on Schölkopf et. al.
            svm = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)

            # fit svm on training data using feature vectors (without any novelties)
            oc_svm = svm.fit(features_train)
            svm_pred = oc_svm.predict(features_train)
            svm_pred = [0 if (i >= 0) else 1 for i in svm_pred]
            
            # construct paths
            model_dir = Path("models") / session_path
            model_fname = Path(f"nu={nu:.6f}_gamm={gamma:.6f}.pkl")
            # make directory and save model
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(oc_svm, model_dir / model_fname)

    return 


if __name__ == '__main__':
    
    from polycraft_nov_data.image_transforms import VGGPreprocess
    from polycraft_nov_data.dataloader import novelcraft_dataloader, novelcraft_plus_dataloader
    from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save
    
    use_novelcraft_plus = True
    evaluation = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    if use_novelcraft_plus:
        train_loader = novelcraft_plus_dataloader("train", VGGPreprocess(), 1)
    else:
        train_loader = novelcraft_dataloader("train", VGGPreprocess(), 1)
    valid_loader = novelcraft_dataloader("valid_norm", VGGPreprocess(), 1)
    test_loader = novelcraft_dataloader("test", VGGPreprocess(), 1)

    output_parent = Path("models/vgg/eval_ocsvm/plus/subset_10000")
    classifier_path = Path("models/vgg/vgg_classifier_1000_plus.pt")
    feature_extractor = load_vgg_model(classifier_path, device).to(device).eval()
    
    nus = np.linspace(0.1, 0.9, 9)
    gammas = np.logspace(-5, 1, 9)
    
    if evaluation == False:
        fit_ocsvm(feature_extractor, train_loader, valid_loader, nus, gammas, device)

    if evaluation:
        
        for nu in nus:
            for gamma in gammas:
                print('gamma: ', gamma, ' nu: ', nu, flush=True)
                output_folder = output_parent / Path(f"nu={nu:.6f}_gamm={gamma:.6f}")
                model_path = output_parent /  Path(f"nu={nu:.6f}_gamm={gamma:.6f}.pkl")
    
                save_scores(
                    OneClassSVMDetector(model_path, classifier_path, device),
                    output_folder,
                    novelcraft_dataloader("valid", VGGPreprocess(), 32),
                    novelcraft_dataloader("test", VGGPreprocess(), 32))
                eval_from_save(output_folder)