import pytest
import torch

from polycraft_nov_data.dataloader import polycraft_dataloaders

from polycraft_nov_det.models.osrnet import OSRNet, OSRNetCNN, OSRNetCS


cifar_input_shape = (3, 32, 32)
batch_size = 256


class TestPolycraft():
    @pytest.fixture
    def model(self):
        return OSRNet(OSRNetCNN(1), OSRNetCS())

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
