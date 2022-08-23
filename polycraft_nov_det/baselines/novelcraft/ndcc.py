import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.dataloader import novelcraft_dataloader, novelcraft_plus_dataloader
from polycraft_nov_data.image_transforms import VGGPreprocess
import polycraft_nov_data.novelcraft_const as nc_const

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.models.vgg import VGGPretrained
import polycraft_nov_det.train as train


class NDCC(nn.Module):
    def __init__(self, embedding, classifier, r=1.0):
        super(NDCC, self).__init__()
        self.embedding = embedding
        self.classifier = classifier
        self.dim_embedding = self.classifier.in_features
        self.num_classes = self.classifier.out_features
        self.r = r

        self.sigma = torch.tensor(((np.ones(1))).astype(
            'float32'), requires_grad=True)
        self.delta = torch.tensor((np.zeros(self.dim_embedding)).astype(
            'float32'), requires_grad=True)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.r * torch.nn.functional.normalize(x, p=2, dim=1)
        return x


def mahalanobis_metric(x, y, inv_sigma):
    # x: numpy array of shape (n_samples_x, n_features)
    # y: numpy array of shape (n_samples_y, n_features)
    # inv_sigma: inverse of sigma (diagonal covariance)
    x_normalized = x * np.sqrt(np.diag(inv_sigma))
    y_normalized = y * np.sqrt(np.diag(inv_sigma))
    distances = pairwise_distances(
        x_normalized, Y=y_normalized, metric='sqeuclidean', n_jobs=1)
    return distances


class NDCCDetector(NoveltyDetector):
    def __init__(self, model: NDCC, device="cpu"):
        super().__init__(device)
        self.model = model.eval().to(device)
        # init Gaussian params
        weight = model.classifier.weight
        weight = weight.cpu().detach().numpy()
        sigma2 = ((model.delta + model.sigma).detach().cpu().numpy()) ** 2
        sigma = np.diag(sigma2)
        self.inv_sigma = np.diag(sigma2 ** -1)
        self.means = weight @ sigma

    def novelty_score(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            outputs = self.model(data)
            outputs = outputs.detach().cpu().numpy().squeeze()
        distances = mahalanobis_metric(outputs, self.means, self.inv_sigma)
        nd_scores = np.min(distances, axis=1)
        return torch.Tensor(nd_scores)


def train_ndcc(model, optimizer, scheduler, num_epochs=20, gpu=None):
    device = torch.device(gpu if gpu is not None else "cpu")
    model.to(device)
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    session_path = Path("NDCC") / Path(start_time)
    # get Tensorboard writer
    writer = SummaryWriter(Path("runs") / session_path)

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.eval()  # NDCC is alwayes set to evaluate mode
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = (inputs.to(device))
                labels = (labels.long().to(device))
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    sigma2 = ((model.sigma + model.delta) ** 2).to(device)
                    means = model.classifier.weight * sigma2
                    loss_md = (torch.div((outputs - means[labels]) ** 2, sigma2.detach())).sum() \
                        / (2 * outputs.shape[0])
                    loss_nll = (torch.log(sigma2).sum()) / 2 + \
                        (torch.div((outputs.detach() - means[labels]) ** 2, sigma2).sum()
                         / outputs.shape[0]) / 2

                    logits = model.classifier(outputs)
                    loss_ce = F.cross_entropy(logits, labels)

                    loss = loss_ce + opt.lmd * (loss_md + opt.gma * loss_nll)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                _, preds = torch.max(logits, 1)
                epoch_loss = (loss.item() * inputs.size(0) +
                              cnt * epoch_loss) / (cnt + inputs.size(0))
                epoch_acc = (torch.sum(preds == labels.data) +
                             epoch_acc * cnt).double() / (cnt + inputs.size(0))
                cnt += inputs.size(0)
            if phase == 'train':
                scheduler.step()
                writer.add_scalar("Average Train Loss", epoch_loss, epoch)
                writer.add_scalar("Average Train Acc", epoch_acc, epoch)
            else:
                writer.add_scalar("Average Valid Loss", epoch_loss, epoch)
                writer.add_scalar("Average Valid Acc", epoch_acc, epoch)
    train.save_model(model, session_path, num_epochs - 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='evaluate instead of training')

    # smaller batch size compared to paper due to limited GPU
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--dataset', type=str, default='StanfordDogsTimes1e-1',
                        choices=['StanfordDogs', 'StanfordDogsTimes1e-1', 'StanfordDogsTimes1e-2'])

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='the number of training epochs')

    parser.add_argument('--lr1', type=float, default=1e-3,
                        help='learning rate for embedding v(x)')
    parser.add_argument('--lr2', type=float, default=1e-1,
                        help='learning rate for linear classifier {w_y, b_y}')
    parser.add_argument('--lr3', type=float, default=1e-1,
                        help='learning rate for sigma')
    parser.add_argument('--lr4', type=float, default=1e-3,
                        help='learning rate for delta_j')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_milestones', default=[5, 10])

    parser.add_argument('--lmd', type=float, default=2e-1,
                        help='lambda in Eq. (23)')
    parser.add_argument('--gma', type=float, default=1 /
                        4096, help='gamma in Eq. (22)')
    parser.add_argument('--r', type=float, default=16, help='|v(x)|=r')

    opt = parser.parse_args()

    if opt.dataset == 'StanfordDogs':
        opt.num_classes = 60
        opt.lr1 = 1e-3
        opt.lr2 = 1e-1
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 16
        opt.lmd = 2e-1

        opt.lr_milestones = [25, 28, 30]
        opt.num_epochs = 30
    elif opt.dataset == 'StanfordDogsTimes1e-1':
        opt.num_classes = 60
        opt.lr1 = 1e-4
        opt.lr2 = 1e-2
        opt.lr3 = 1e-2
        opt.lr4 = 1e-4
        opt.r = 16
        opt.lmd = 2e-1

        opt.lr_milestones = [25, 28, 30]
        opt.num_epochs = 30
    elif opt.dataset == 'StanfordDogsTimes1e-2':
        opt.num_classes = 60
        opt.lr1 = 1e-5
        opt.lr2 = 1e-3
        opt.lr3 = 1e-3
        opt.lr4 = 1e-5
        opt.r = 16
        opt.lmd = 2e-1

        opt.lr_milestones = [25, 28, 30]
        opt.num_epochs = 30

    use_novelcraft_plus = True
    if use_novelcraft_plus:
        train_loader = novelcraft_plus_dataloader("train", VGGPreprocess(), opt.batch_size,
                                                  balance_classes=True)
    else:
        train_loader = novelcraft_dataloader("train", VGGPreprocess(), opt.batch_size)
    valid_loader = novelcraft_dataloader("valid_norm", VGGPreprocess(), opt.batch_size)
    test_loader = novelcraft_dataloader("test", VGGPreprocess(), opt.batch_size)
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = valid_loader

    embedding = VGGPretrained(len(nc_const.NORMAL_CLASSES)).backbone  # num_classes ignored here
    classifier = nn.Linear(4096, len(nc_const.NORMAL_CLASSES))  # replacing typical VGG head
    model = NDCC(embedding=embedding, classifier=classifier, r=opt.r)

    optimizer = torch.optim.SGD([{'params': model.embedding.parameters(), 'lr': opt.lr1},
                                 {'params': model.classifier.parameters(
                                 ), 'lr': opt.lr2, 'weight_decay': 0e-4},
                                 {'params': [
                                  model.sigma], 'lr': opt.lr3, 'weight_decay': 0e-4},
                                 {'params': [
                                  model.delta], 'lr': opt.lr4, 'weight_decay': 0e-4},
                                 ], momentum=opt.momentum, weight_decay=5e-4)

    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.lr_milestones, gamma=0.1)

    device = torch.device("cuda:1")
    # train model
    if not opt.eval:
        train_ndcc(model, optimizer, scheduler, num_epochs=opt.num_epochs, gpu=device)
    # eval model
    else:
        from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save
        from polycraft_nov_det.model_utils import load_model

        model = load_model("models/vgg/ndcc_plus_30.pt", model, device)
        output_folder = Path("models/vgg/eval_ndcc/plus/")
        save_scores(NDCCDetector(model, device), output_folder, valid_loader, test_loader)
        eval_from_save(output_folder)
