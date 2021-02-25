from importlib.resources import path

from torch.utils import data
from torchvision.datasets import MNIST
from torchvision import transforms


def torch_mnist(batch_size=1, shuffle=True):
    """torch DataLoaders for MNIST

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 1.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.

    Returns:
        (DataLoader, DataLoader): MNIST train and test sets
    """
    # get path within module
    with path("polycraft_nov_det", "base_data") as data_path:
        # get dataset with separate data and targets
        train_set = MNIST(root=data_path, train=True, download=True,
                          target_transform=transforms.ToTensor())
        test_set = MNIST(root=data_path, train=False, download=True,
                         target_transform=transforms.ToTensor())
    # adds extra channel dimension to data for compatibility with networks expecting RGB
    train_data = train_set.data[:, None].float()
    test_data = test_set.data[:, None].float()
    # get tensor dataset with data and targets
    train_tensor = data.TensorDataset(train_data, train_set.targets)
    test_tensor = data.TensorDataset(test_data, test_set.targets)
    # get DataLoaders for datasets
    return (data.DataLoader(train_tensor, batch_size, shuffle),
            data.DataLoader(test_tensor, batch_size, shuffle))
