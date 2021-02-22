from importlib.resources import path

import torchvision.datasets


def torch_mnist():
    """torchvision datasets for MNIST

    Returns:
        (torchvision.datasets.mnist.MNIST,
         torchvision.datasets.mnist.MNIST): MNIST train and test sets
    """
    # get path within module
    with path("polycraft_nov_det", "base_data") as data_path:
        train = torchvision.datasets.MNIST(root=data_path, train=True, download=True)
        test = torchvision.datasets.MNIST(root=data_path, train=False, download=True)
    return train, test
