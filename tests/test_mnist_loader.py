import polycraft_nov_det.mnist_loader as mnist_loader


def test_mnist_len():
    # check length of datasets
    train_loader, valid_loader, test_loader = mnist_loader.torch_mnist(batch_size=1, shuffle=False)
    assert len(train_loader) + len(valid_loader) == 30596
    assert len(test_loader) == 5139
    train_loader, valid_loader, test_loader = mnist_loader.torch_mnist(
        batch_size=1, include_novel=True, shuffle=False)
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000
