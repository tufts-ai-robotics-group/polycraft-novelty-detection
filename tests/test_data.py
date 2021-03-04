import polycraft_nov_det.data as data


# MNIST tests
def test_mnist():
    train_loader, valid_loader, test_loader = data.torch_mnist()
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000
