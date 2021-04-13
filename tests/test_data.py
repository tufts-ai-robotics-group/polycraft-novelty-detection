import polycraft_nov_det.data as data


# MNIST tests
train_loader, valid_loader, test_loader = data.torch_mnist(batch_size=1, shuffle=False)


def test_mnist_len():
    # check length of datasets
    assert len(train_loader) + len(valid_loader) == 60000
    assert len(test_loader) == 10000


# GaussianNoise tests
def test_gaussian_noise():
    # check bounds of noisy image and that noise is applied
    image = next(train_loader.__iter__())[0]
    noisy_image = data.GaussianNoise()(image)
    assert noisy_image.min() >= 0
    assert noisy_image.max() <= 1
    assert (noisy_image != image).any()
