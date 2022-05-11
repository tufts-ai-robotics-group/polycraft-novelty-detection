import argparse

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.models.vgg import VGGPretrained


class NDCC(nn.Module):
    def __init__(self, embedding, classifier, r=1.0):
        super(NDCC, self).__init__()
        self.embedding = embedding
        self.classifier = classifier
        self.dim_embedding = self.classifier.in_features
        self.num_classes = self.classifier.out_features
        self.r = r

        self.sigma = torch.tensor(((np.ones(1))).astype(
            'float32'), requires_grad=True, device="cuda")
        self.delta = torch.tensor((np.zeros(self.dim_embedding)).astype(
            'float32'), requires_grad=True, device="cuda")

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
        return nd_scores


def train_ndcc(model, optimizer, scheduler, num_epochs=20):
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.eval()  # NDCC is alwayes set to evaluate mode
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = (inputs.cuda())
                labels = (labels.long().cuda())
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    sigma2 = (model.sigma + model.delta) ** 2
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
    torch.save(model.state_dict(), saved_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)

    parser.add_argument('--dataset', type=str, default='FounderType200',
                        choices=['CUB200', 'StanfordDogs', 'FounderType200'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed for train/test split')

    parser.add_argument('--num_classes', type=int, default=100,
                        help='the number of training classes')
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

    parser.add_argument('--exp_id', type=str, default='1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    opt = parser.parse_args()

    # recommended choice for hyperparameters (according to Table C.1. in our Supplementary Material)
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

    elif opt.dataset == 'FounderType200':
        opt.num_classes = 100
        opt.lr1 = 1e-2
        opt.lr2 = 1e-1
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 32
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 10

    elif opt.dataset == 'CUB200':
        opt.num_classes = 100
        opt.lr1 = 1e-2
        opt.lr2 = 1e-1
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 8
        opt.lmd = 1e-4

        opt.lr_milestones = [30, 40]
        opt.num_epochs = 40

    embedding = VGGPretrained(5).backbone  # num_classes ignored here
    classifier = nn.Linear(4096, opt.num_classes)
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

    model = model.cuda()

    # ==================== training ====================

    train_ndcc(model, optimizer, scheduler, num_epochs=opt.num_epochs)
