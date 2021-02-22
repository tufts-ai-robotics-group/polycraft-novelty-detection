import torch

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst


input_shape = (3, 32, 32)
batch_size = 256


# constructor tests
def test_constructor_no_est():
    LSACIFAR10NoEst(input_shape, 64)


# forward tests
def test_forward_no_est():
    model = LSACIFAR10NoEst(input_shape, 64)
    model(torch.zeros((batch_size,) + input_shape))
