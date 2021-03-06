import argparse

from polycraft_nov_data.data_const import PATCH_SHAPE
from polycraft_nov_data.dataloader import polycraft_dataloaders

import polycraft_nov_det.mini_imagenet_loader as mini_imagenet_loader
import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.models.lsa.LSA_cifar10_no_est as LSA_cifar10_no_est
import polycraft_nov_det.models.lsa.LSA_mnist_no_est as LSA_mnist_no_est
import polycraft_nov_det.train as train


# construct argument parser
parser = argparse.ArgumentParser(description="Polycraft Novelty Detection Model Training")
parser.add_argument("model", choices=["mnist", "polycraft", "miniimagenet"],
                    help="Model to train")
# Polycraft specific args
polycraft_group = parser.add_argument_group("Polycraft")
polycraft_group.add_argument("-image_scale", type=float,
                             help="float in (0, 1] to scale Polycraft images by")
# data specific args
data_group = parser.add_argument_group("data")
data_group.add_argument("-batch_size", type=int,
                        help="Batch size for DataLoader")
# model specific args
model_group = parser.add_argument_group("model")
model_group.add_argument("-latent_len", type=int,
                         help="Dimensionality of latent space")
# training specific args
training_group = parser.add_argument_group("training")
training_group.add_argument("-epochs", type=int,
                            help="Number of epochs to train for")
training_group.add_argument("-lr", type=float,
                            help="Learning rate")
training_group.add_argument("-no_noise", action="store_true",
                            help="Don't add noise to training images")
training_group.add_argument("-gpu", type=int,
                            help="Index of GPU to train on, negative int for CPU")
args = parser.parse_args()

include_classes = None
# handle MNIST args
if args.model == "mnist":
    # set default train kwargs
    train_kwargs = {
        "lr": 1e-2,
        "epochs": 500,
        "gpu": 1,
    }
    # get dataloaders
    batch_size = 256 if args.batch_size is None else args.batch_size
    train_loader, valid_loader, _ = mnist_loader.torch_mnist(batch_size)
    # get model instance
    latent_len = 64 if args.latent_len is None else args.latent_len
    model = LSA_mnist_no_est.LSAMNISTNoEst(mnist_loader.MNIST_SHAPE, latent_len)
# handle Polycraft args
elif args.model == "polycraft":
    # set default train kwargs
    train_kwargs = {
        "lr": 1e-3,
        "epochs": 8000,
        "gpu": 1,
    }
    # get dataloaders
    batch_size = 128 if args.batch_size is None else args.batch_size
    image_scale = 1.0 if args.image_scale is None else args.image_scale
    train_loader, valid_loader, _ = polycraft_dataloaders(batch_size, image_scale)
    # get model instance
    latent_len = 100 if args.latent_len is None else args.latent_len
    model = LSA_cifar10_no_est.LSACIFAR10NoEst(PATCH_SHAPE, latent_len)
# handle Mini-ImageNet args
elif args.model == "miniimagenet":
    # set default train kwargs
    train_kwargs = {
        "lr": 1e-3,
        "epochs": 1000,
        "gpu": 1,
    }
    # get dataloaders
    batch_size = 128 if args.batch_size is None else args.batch_size
    train_loader, valid_loader, _ = mini_imagenet_loader.mini_imagenet_dataloaders(batch_size)
    # get model instance
    latent_len = 100 if args.latent_len is None else args.latent_len
    model = LSA_cifar10_no_est.LSACIFAR10NoEst(mini_imagenet_loader.PATCH_SHAPE, latent_len)


# update train_kwargs with parsed args
args_dict = vars(args)
train_kwargs = {key: val if args_dict[key] is None else args_dict[key]
                for key, val in train_kwargs.items()}
# use CPU if negative GPU index given
if train_kwargs["gpu"] < 0:
    train_kwargs["gpu"] = None
# start model training
model_label = train.model_label(model, include_classes)
train.train(model, model_label, train_loader, valid_loader, train_noisy=not args.no_noise,
            **train_kwargs)
