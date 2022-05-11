import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn


class NDCC(nn.Module):
    def __init__(self, embedding, classifier, strategy, l2_normalize=True, r=1.0):
        super(NDCC, self).__init__()
        self.embedding = embedding
        self.classifier = classifier
        self.dim_embedding = self.classifier.in_features
        self.num_classes = self.classifier.out_features
        self.l2_normalize = l2_normalize
        if self.l2_normalize:
            self.r = r

        self.strategy = strategy
        if self.strategy == 1:
            self.sigma = torch.tensor(((np.ones(1))).astype(
                'float32'), requires_grad=True, device="cuda")
            self.delta = None
        elif self.strategy == 2:
            self.sigma = torch.tensor(((np.ones(1))).astype(
                'float32'), requires_grad=True, device="cuda")
            self.delta = torch.tensor((np.zeros(self.dim_embedding)).astype(
                'float32'), requires_grad=True, device="cuda")

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        if self.l2_normalize:
            x = self.r * torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def get_nd_scores(self, loader):
        self.eval()
        weight = self.classifier.weight
        weight = weight.cpu().detach().numpy()
        if self.strategy == 1:
            sigma2 = (self.sigma.detach().cpu().numpy()) ** 2
            sigma = sigma2 * np.eye(self.dim_embedding)
            inv_sigma = (1/sigma2) * np.eye(self.dim_embedding)
        elif self.strategy == 2:
            sigma2 = ((self.delta + self.sigma).detach().cpu().numpy()) ** 2
            sigma = np.diag(sigma2)
            inv_sigma = np.diag(sigma2 ** -1)
        means = weight @ sigma
        distances = np.zeros((len(loader.dataset), self.num_classes))
        with torch.no_grad():
            idx = 0
            # Iterate over data.
            for step, (inputs, _) in enumerate(loader):
                if step % 100 == 0:
                    print(step)
                inputs = inputs.cuda()
                outputs = nn.parallel.data_parallel(self, inputs)
                outputs = outputs.detach().cpu().numpy().squeeze()
                distances[idx:idx+len(outputs),
                          :] = mahalanobis_metric(outputs, means, inv_sigma)
                idx += len(outputs)
        nd_scores = np.min(distances, axis=1)
        return nd_scores


def mahalanobis_metric(x, y, inv_sigma):
    # x: numpy array of shape (n_samples_x, n_features)
    # y: numpy array of shape (n_samples_y, n_features)
    # inv_sigma: inverse of sigma (diagonal covariance)
    x_normalized = x * np.sqrt(np.diag(inv_sigma))
    y_normalized = y * np.sqrt(np.diag(inv_sigma))
    distances = pairwise_distances(
        x_normalized, Y=y_normalized, metric='sqeuclidean', n_jobs=1)
    return distances
