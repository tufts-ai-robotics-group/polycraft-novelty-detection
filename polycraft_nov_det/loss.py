import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy


class SimilarBCE(nn.Module):
    def is_similar(self, norm_desc_1, norm_desc_2, k=5):
        """Determine if two embeddings have same top K ranking

        Args:
            norm_desc_1 (torch.Tensor): Descriptor for normal classes (B, D).
            norm_desc_2 (torch.Tensor): Descriptor for normal classes (B, D).
            k (int, optional): Number of dimensions to match. Defaults to 5.

        Returns:
            torch.Tensor: Return 1 if top K dimensions are the same, otherwise 0. (B,)
        """
        # get top K dimensions
        top_k_1 = torch.argsort(norm_desc_1, dim=1, descending=True)[:, :k]
        top_k_2 = torch.argsort(norm_desc_2, dim=1, descending=True)[:, :k]
        # compute which sets are equal
        top_k_dif = torch.abs(torch.sort(top_k_1, dim=1)[0] - torch.sort(top_k_2, dim=1)[0])
        top_k_dif = torch.sum(top_k_dif, dim=1)
        # assign 1 if sets are equal
        similar_label = torch.zeros_like(top_k_dif)
        similar_label[top_k_dif == 0] = 1
        return similar_label

    def forward(self, norm_desc_1, norm_desc_2, nov_desc_1, nov_desc_2):
        prod = nov_desc_1 @ nov_desc_2.T
        similar_label = self.is_similar(norm_desc_1, norm_desc_2)
        return binary_cross_entropy(prod, similar_label)
