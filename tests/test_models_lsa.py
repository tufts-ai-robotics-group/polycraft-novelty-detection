import pytest
import torch

from polycraft_nov_data.dataloader import novelcraft_dataloader
from polycraft_nov_data.image_transforms import PatchTrainPreprocess

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst


cifar_input_shape = (3, 32, 32)
batch_size = 256


class TestCIFAR10():
    @pytest.fixture
    def model(self):
        return LSACIFAR10NoEst(cifar_input_shape, 64)

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + cifar_input_shape))

    def test_novelcraft_dataloader(self, model):
        train_loader = novelcraft_dataloader("train", PatchTrainPreprocess(), batch_size=batch_size)
        for data, _ in train_loader:
            model(data)
            break
