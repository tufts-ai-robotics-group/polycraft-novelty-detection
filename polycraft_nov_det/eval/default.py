import polycraft_nov_det.eval.evals as evals
from polycraft_nov_det.model_load import load_autonovel_resnet, load_autonovel_pretrained, \
    load_dino_block, load_dino_pretrained


# pretrained models


def cifar10_self_supervised_pretrained():
    evals.cifar10_self_supervised(
        load_autonovel_pretrained("models/AutoNovel/self_supervised/rotnet_cifar10.pth", 4, 0))


def cifar10_supervised_pretrained():
    evals.cifar10_supervised(
        load_autonovel_pretrained("models/AutoNovel/supervised/resnet_rotnet_cifar10.pth", 5, 0))


def cifar10_autonovel_pretrained():
    evals.cifar10_autonovel(
        load_autonovel_pretrained("models/AutoNovel/discovery/resnet_IL_cifar10.pth", 10, 5))


def cifar10_gcd_pretrained():
    evals.cifar10_gcd(load_dino_pretrained())


# default trained models


def cifar10_self_supervised():
    evals.cifar10_self_supervised(
        load_autonovel_resnet("models/CIFAR10/self_supervised/200.pt", 4, 0))


def cifar10_supervised():
    evals.cifar10_supervised(
        load_autonovel_resnet("models/CIFAR10/supervised/100.pt", 5, 0))


def cifar10_autonovel():
    evals.cifar10_autonovel(
        load_autonovel_resnet("models/CIFAR10/discovery/200.pt", 10, 5))


def cifar10_gcd():
    evals.cifar10_gcd(load_dino_block("models/CIFAR10/GCD/block200.pt"))
