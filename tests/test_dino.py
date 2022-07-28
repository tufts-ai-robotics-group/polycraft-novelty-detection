import pytest
import torch

from polycraft_nov_data.novelcraft_const import IMAGE_SHAPE

from polycraft_nov_det.models.dino_train import DinoWithHead
from polycraft_nov_det.model_load import load_dino_pretrained


batch_size = 64


class TestDino():
    @pytest.fixture
    def model(self):
        return DinoWithHead(load_dino_pretrained())

    def test_forward(self, model):
        model(torch.zeros((batch_size,) + IMAGE_SHAPE))
