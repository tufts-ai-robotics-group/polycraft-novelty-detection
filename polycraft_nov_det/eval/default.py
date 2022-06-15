import polycraft_nov_det.eval.evals as evals
from polycraft_nov_det.model_load import load_dino_block, load_dino_pretrained


# pretrained models


def cifar10_gcd_pretrained():
    evals.cifar10_gcd(load_dino_pretrained())


# default trained models


def cifar10_gcd():
    evals.cifar10_gcd(load_dino_block("models/CIFAR10/GCD/block200.pt"))
