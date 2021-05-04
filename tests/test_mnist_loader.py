import polycraft_nov_det.mnist_loader as mnist_loader


# MNIST tests
train_loader, valid_loader, test_loader = mnist_loader.torch_mnist(batch_size=1, shuffle=False)


def test_mnist_len():
    # check length of datasets
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000
