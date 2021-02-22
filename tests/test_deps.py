# test torch and torchvision have been installed, since they're separate from the Pipenv
def test_torch():
    try:
        import torch
    except Exception as e:
        print(e)
        raise Exception("PyTorch not installed correctly")


def test_torchvision():
    try:
        import torchvision
    except Exception as e:
        print(e)
        raise Exception("torchvision not installed correctly")
