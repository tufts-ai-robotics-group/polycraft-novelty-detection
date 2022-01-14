import pytest
import torch

import polycraft_nov_det.data.cifar_loader as cifar_loader
from polycraft_nov_det.models.disc_resnet import DiscResNet


batch_size = 64


class TestCIFAR10():
    @pytest.fixture
    def model(self):
        return DiscResNet(5, 5)

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + cifar_loader.CIFAR_SHAPE))

    def test_dataloader_cifar(self, model):
        norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
            batch_size=batch_size, shuffle=False)
        train_loader, valid_loader, test_loader = dataloaders
        for data, targets in train_loader:
            model(data)
            break

    def test_dataloader_rot_cifar(self, model):
        norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
            batch_size=batch_size, shuffle=False, rot_loader="rotnet")
        train_loader, valid_loader, test_loader = dataloaders
        for data, targets in train_loader:
            model(data)
            break

    def test_dataloader_rot_const_cifar(self, model):
        norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
            batch_size=batch_size, shuffle=False, rot_loader="consistent")
        train_loader, valid_loader, test_loader = dataloaders
        for data, rot_data, targets in train_loader:
            model(data)
            model(rot_data)
            break
