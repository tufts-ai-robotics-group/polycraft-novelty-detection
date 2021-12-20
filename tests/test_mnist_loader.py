import polycraft_nov_det.data.mnist_loader as mnist_loader


def test_mnist_len():
    # check length of datasets
    norm_targets, novel_targets, dataloaders = mnist_loader.torch_mnist(
        batch_size=1, shuffle=False)
    train_loader, valid_loader, test_loader = dataloaders
    assert len(train_loader) + len(valid_loader) == 30311
    assert len(test_loader) == 5081
    norm_targets, novel_targets, dataloaders = mnist_loader.torch_mnist(
        batch_size=1, include_novel=True, shuffle=False)
    train_loader, valid_loader, test_loader = dataloaders
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000
