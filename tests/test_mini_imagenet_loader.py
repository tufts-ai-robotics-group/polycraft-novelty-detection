from polycraft_nov_det.mini_imagenet_loader import mini_imagenet_dataloaders


# Mini-ImageNet tests
train_loader, valid_loader, test_loader = mini_imagenet_dataloaders(batch_size=1, shuffle=False)


def test_mini_imagenet_len():
    # check length of datasets
    assert len(train_loader) == 38400
    assert len(valid_loader) == 9600
    assert len(test_loader) == 12000
