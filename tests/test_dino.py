import pytest
import torch

import polycraft_nov_det.data.cifar_loader as cifar_loader
from polycraft_nov_det.models.dino_train import DinoWithHead
from polycraft_nov_det.model_load import load_dino_pretrained


batch_size = 64


class TestCIFAR10():
    @pytest.fixture
    def model(self):
        return DinoWithHead(load_dino_pretrained())

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + cifar_loader.CIFAR_SHAPE))

    def test_dataloader_cifar(self, model):
        norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
            range(5), batch_size=batch_size, shuffle=False)
        train_loader, valid_loader, test_loader = dataloaders
        for data, targets in train_loader:
            out = model(data)
            break
