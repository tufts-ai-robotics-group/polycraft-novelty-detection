import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarBCE(nn.Module):
    def is_similar(self, unlabel_feat, k=5):
        """Determine if two embeddings have same top K ranking

        Args:
            unlabel_feat (torch.Tensor): Feature for unlabeled data (B, D).
            k (int, optional): Number of dimensions to match. Defaults to 5.

        Returns:
            torch.Tensor: Return 1 if top K dimensions are the same, otherwise 0.
                          Computes all pairings, so shape is (B, B).
        """
        batch_size = unlabel_feat.shape[0]
        # get top K dimensions and sort for later comparisons
        top_k = torch.topk(unlabel_feat, k, dim=1)[1]
        # assign 1 if sets are equal, computing with same indices as a matrix multiplication
        similar_label = torch.zeros((batch_size, batch_size))
        for i in range(batch_size):
            similar_label[i, torch.all(top_k[i] == top_k, dim=1)] = 1
        return similar_label

    def forward(self, unlabel_feat, unlabel_prob, rot_unlabel_prob):
        # dot product of image pairs with different transforms
        prod = unlabel_prob @ rot_unlabel_prob.T
        # get similarity labels for pairs of features of unlabeled images
        similar_label = self.is_similar(unlabel_feat)
        return F.binary_cross_entropy(prod, similar_label)


class AutoNovelLoss(nn.Module):
    def __init__(self, norm_targets, consist_coef=50, il_coef=.05, ramp_len=150):
        super().__init__()
        self.consist_coef = consist_coef
        self.il_coef = il_coef
        self.ramp_len = ramp_len
        self.num_norm_targets = len(norm_targets)
        self.sim_bce = SimilarBCE()

    def gaus_ramp(self, val, epoch):
        # Gaussian ramp from https://arxiv.org/abs/1610.02242
        ramp_scaling = 1
        if self.ramp_len > 0:
            ramp_input = torch.tensor(min(epoch, self.ramp_len) / self.ramp_len)
            ramp_scaling = torch.exp(-5 * (1 - ramp_input) ** 2)
        return ramp_scaling * val

    def forward(self, label_pred, unlabel_pred, feat, rot_label_pred, rot_unlabel_pred,
                targets, epoch):
        # select labeled images according to the class reordering of base_dataset
        norm_mask = targets < self.num_norm_targets
        # calculate cross entropy loss on normal labels
        loss_ce = F.cross_entropy(label_pred[norm_mask], targets[norm_mask])
        # calculate cross entropy loss on psuedo labels for unlabeled images
        il_psuedo_targets = torch.max(unlabel_pred[~norm_mask], 1)[1] + self.num_norm_targets
        loss_il_ce = F.cross_entropy(label_pred[~norm_mask], il_psuedo_targets)
        # calculate consistency loss between unrotated and rotated
        label_prob = F.softmax(label_pred, dim=1)
        rot_label_prob = F.softmax(rot_label_pred, dim=1)
        unlabel_prob = F.softmax(unlabel_pred, dim=1)
        rot_unlabel_prob = F.softmax(rot_unlabel_pred, dim=1)
        loss_consistency = F.mse_loss(label_prob, rot_label_prob) + \
            F.mse_loss(unlabel_prob, rot_unlabel_prob)
        # calculate binary cross entropy loss on unlabeled images with psuedo labels
        loss_bce = self.sim_bce(feat[~norm_mask], unlabel_prob[~norm_mask],
                                rot_unlabel_prob[~norm_mask])
        # calculate ramped weights
        w_consistency = self.gaus_ramp(self.consist_coef, epoch)
        w_il = self.gaus_ramp(self.il_coef, epoch)
        return loss_ce + loss_bce + w_il * loss_il_ce + w_consistency * loss_consistency
