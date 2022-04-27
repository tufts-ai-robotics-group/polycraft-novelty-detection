import pytest
import torch

import polycraft_nov_det.models.vgg as vgg


batch_size = 64


class TestVGG():
    @pytest.fixture
    def model(self):
        return vgg.VGGPretrained(3)

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + (3, 256, 256 - 22)))
