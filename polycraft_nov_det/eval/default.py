import polycraft_nov_det.eval.evals as evals
from polycraft_nov_det.model_utils import load_autonovel_resnet, load_autonovel_pretrained


# pretrained models


def cifar10_self_supervised_pretrained():
    evals.cifar10_self_supervised(
        load_autonovel_pretrained("models/AutoNovel/self_supervised/rotnet_cifar10.pth", 4, 0))


def cifar10_clustering_pretrained():
    evals.cifar10_clustering(
        load_autonovel_pretrained("models/AutoNovel/discovery/resnet_IL_cifar10.pth", 10, 5))


# default trained models


def cifar10_self_supervised():
    evals.cifar10_self_supervised(
        load_autonovel_resnet("models/CIFAR10/self_supervised/200.pt", 4, 0))


def cifar10_clustering():
    evals.cifar10_clustering(
        load_autonovel_resnet("models/CIFAR10/discovery/200.pt", 10, 5))
