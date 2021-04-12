import polycraft_nov_det.data as data


# MNIST tests
train_loader, valid_loader, test_loader = data.torch_mnist(batch_size=1, shuffle=False)


def test_mnist_len():
    # check length of datasets
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000


def test_mnist_noise():
    # check bounds of noisy image and that noise is applied
    noisy_loader, _, _ = data.torch_mnist(batch_size=1, shuffle=False, noise=True)
    image = next(train_loader.__iter__())[0]
    noisy_image = next(noisy_loader.__iter__())[0]
    assert noisy_image.min() >= 0
    assert noisy_image.max() <= 1
    assert (noisy_image != image).any()
