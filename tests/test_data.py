import polycraft_nov_det.data as data


# MNIST tests
def test_mnist():
    train, test = data.torch_mnist()
    assert len(train) == 60000
    assert len(test) == 10000
