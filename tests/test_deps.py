# test torch and torchvision have been installed, since they're separate from the Pipenv
def test_torch():
    try:
        import torch
        torch.zeros(10)
    except Exception as e:
        print(e)
        raise Exception("PyTorch not installed correctly")


def test_torchvision():
    try:
        import torchvision
        torchvision.transforms
    except Exception as e:
        print(e)
        raise Exception("torchvision not installed correctly")
