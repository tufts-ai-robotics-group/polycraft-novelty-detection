import polycraft_nov_det.data.cifar_loader as cifar_loader


def test_cifar10_len():
    # check length of datasets
    norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
        range(5), batch_size=1, shuffle=False)
    train_loader, valid_loader, test_loader = dataloaders
    assert len(train_loader) + len(valid_loader) == 25000
    assert len(test_loader) == 5000
    norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
        range(5), batch_size=1, include_novel=True, shuffle=False)
    train_loader, valid_loader, test_loader = dataloaders
    assert len(train_loader) + len(valid_loader) == 50000
    assert len(test_loader) == 10000


def test_cifar100_len():
    # check length of datasets
    norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
        range(50), batch_size=1, shuffle=False, use_10=False)
    train_loader, valid_loader, test_loader = dataloaders
    assert len(train_loader) + len(valid_loader) == 25000
    assert len(test_loader) == 5000
    norm_targets, novel_targets, dataloaders = cifar_loader.torch_cifar(
        range(50), batch_size=1, include_novel=True, shuffle=False, use_10=False)
    train_loader, valid_loader, test_loader = dataloaders
    assert len(train_loader) + len(valid_loader) == 50000
    assert len(test_loader) == 10000
