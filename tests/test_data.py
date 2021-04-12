import polycraft_nov_det.data as data


# MNIST tests
train_loader, valid_loader, test_loader = data.torch_mnist(batch_size=1, shuffle=False)


def test_mnist_len():
    # check length of datasets
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000
