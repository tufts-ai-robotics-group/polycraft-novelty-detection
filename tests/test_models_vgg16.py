import torch

import polycraft_nov_det.models.vgg as vgg


batch_size = 8


class TestVGG():
    def test_forward(self):
        model = vgg.VGGPretrained(3)
        model(torch.zeros((batch_size,) + (3, 256, 256 - 22)))
