import argparse

from polycraft_nov_det.data.cifar_loader import torch_cifar
from polycraft_nov_det.models.autonovel_resnet import AutoNovelResNet
from polycraft_nov_det.model_load import load_autonovel_resnet, load_dino_pretrained
from polycraft_nov_det.models.dino_train import DinoWithHead
import polycraft_nov_det.train as train

# construct argument parser
parser = argparse.ArgumentParser(description="Polycraft Novelty Detection Model Training")
# add args
parser.add_argument("-model", choices=["self_supervised", "supervised", "autonovel", "gcd"],
                    default="gcd", help="Model to train")
parser.add_argument("-name", default=None, help="Name for model run")
parser.add_argument("-gpu", type=int, default=1,
                    help="Index of GPU to train on, negative int for CPU")
args = parser.parse_args()
# process args
if args.gpu < 0:
    args.gpu = None

if args.model == "self_supervised":
    # get dataloaders, note that batch is effectively 128 becuase of 4 rotations per image
    batch_size = 32
    _, _, (train_loader, _, _) = torch_cifar(
        range(5), batch_size, include_novel=True, rot_loader="rotnet")
    # get model instance
    model = AutoNovelResNet(4, 0)
    # start model training
    model_label = args.name if args.name is not None else train.model_label(model, range(4))
    train.train_self_supervised(model, model_label, train_loader, gpu=args.gpu)
elif args.model == "supervised":
    # get dataloaders
    batch_size = 128
    norm_targets, _, (train_loader, _, _) = torch_cifar(range(5), batch_size)
    # get model instance
    model = load_autonovel_resnet("models/CIFAR10/self_supervised/200.pt", len(norm_targets), 0,
                                  reset_head=True, strict=False)
    model.freeze_layers()
    # start model training
    model_label = args.name if args.name is not None else train.model_label(model, norm_targets)
    train.train_supervised(model, model_label, train_loader, gpu=args.gpu)
elif args.model == "autonovel":
    # get dataloaders
    # TODO test this should be 64 and not 128 due to transform twice
    batch_size = 64
    norm_targets, novel_targets, (train_loader, _, _) = torch_cifar(
        range(5), batch_size, include_novel=True, rot_loader="consistent")
    # get model instance
    model = load_autonovel_resnet(
        "models/CIFAR10/supervised/100.pt", len(norm_targets), len(novel_targets), reset_head=False,
        strict=False, to_incremental=True)
    # start model training
    model_label = args.name if args.name is not None else train.model_label(model, norm_targets)
    train.train_autonovel(model, model_label, train_loader, norm_targets, gpu=args.gpu)
elif args.model == "gcd":
    batch_size = 128
    norm_targets, novel_targets, (train_loader, _, _) = torch_cifar(
        range(5), batch_size, include_novel=True, rot_loader="consistent")
    # get model instance
    model = DinoWithHead(load_dino_pretrained())
    # start model training
    model_label = args.name if args.name is not None else train.model_label(model, norm_targets)
    train.train_gcd(model, model_label, train_loader, norm_targets, gpu=args.gpu)
