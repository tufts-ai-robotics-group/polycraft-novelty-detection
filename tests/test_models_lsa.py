import torch

from polycraft_nov_det.data import torch_mnist
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst
from polycraft_nov_det.models.lsa.LSA_mnist_no_est import LSAMNISTNoEst


cifar_input_shape = (3, 32, 32)
mnist_input_shape = (1, 28, 28)
batch_size = 256


# constructor tests
def test_constructor_cifar10_no_est():
    LSACIFAR10NoEst(cifar_input_shape, 64)


def test_constructor_mnist_no_est():
    LSAMNISTNoEst(mnist_input_shape, 64)


# forward tests
def test_forward_cifar10_no_est():
    model = LSACIFAR10NoEst(cifar_input_shape, 64)
    model(torch.zeros((batch_size,) + cifar_input_shape))


def test_forward_mnist_no_est():
    model = LSAMNISTNoEst(mnist_input_shape, 64)
    model(torch.zeros((batch_size,) + mnist_input_shape))


def test_mnist_no_est():
    train_loader, _, _ = torch_mnist(batch_size)
    model = LSAMNISTNoEst(mnist_input_shape, 64)
    for data, target in train_loader:
        model(data)
        break
