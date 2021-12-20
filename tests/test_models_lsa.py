import pytest
import torch

from polycraft_nov_data.dataloader import polycraft_dataloaders

from polycraft_nov_det.data.mnist_loader import torch_mnist, MNIST_SHAPE
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst
from polycraft_nov_det.models.lsa.LSA_mnist_no_est import LSAMNISTNoEst


cifar_input_shape = (3, 32, 32)
batch_size = 256


class TestCIFAR10():
    @pytest.fixture
    def model(self):
        return LSACIFAR10NoEst(cifar_input_shape, 64)

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + cifar_input_shape))

    def test_dataloader_polycraft(self, model):
        train_loader, _, _ = polycraft_dataloaders(batch_size)
        patch_train_loader, _, _ = polycraft_dataloaders(include_novel=True)
        for data, _ in train_loader:
            model(data)
            break
        for data, _ in patch_train_loader:
            model(data)
            break


class TestMNIST():
    @pytest.fixture
    def model(self):
        return LSAMNISTNoEst(MNIST_SHAPE, 64)

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + MNIST_SHAPE))

    def test_dataloader_mnist(self, model):
        _, _, dataloaders = torch_mnist(batch_size)
        train_loader, _, _ = dataloaders
        for data, _ in train_loader:
            model(data)
            break
