from polycraft_nov_det.data.cifar_loader import torch_cifar
from polycraft_nov_det.models.disc_resnet import DiscResNet
from polycraft_nov_det.model_utils import load_disc_resnet
import polycraft_nov_det.train as train

mode = "autonovel"

if mode == "self_supervised":
    # get dataloaders
    batch_size = 128
    _, _, (train_loader, valid_loader, _) = torch_cifar(
        batch_size, include_novel=True, rot_loader="rotnet")
    # get model instance
    model = DiscResNet(4, 0)
    # start model training
    model_label = train.model_label(model, range(4))
    train.train_self_supervised(model, model_label, train_loader, valid_loader, gpu=1)
elif mode == "supervised":
    # get dataloaders
    batch_size = 128
    norm_targets, _, (train_loader, valid_loader, _) = torch_cifar(batch_size)
    # get model instance
    model = load_disc_resnet("models/CIFAR10/self_supervised/200.pt", len(norm_targets), 0,
                             reset_head=True, strict=False)
    # start model training
    model_label = train.model_label(model, norm_targets)
    train.train_supervised(model, model_label, train_loader, valid_loader, gpu=1)
elif mode == "autonovel":
    # get dataloaders
    batch_size = 128
    norm_targets, novel_targets, (train_loader, valid_loader, _) = torch_cifar(
        batch_size, include_novel=True, rot_loader="consistent")
    # get model instance
    model = load_disc_resnet("models/CIFAR10/supervised/100.pt", len(norm_targets),
                             len(novel_targets), reset_head=False, strict=False, incremental=True)
    # start model training
    model_label = train.model_label(model, norm_targets)
    train.train_autonovel(model, model_label, train_loader, valid_loader, norm_targets, gpu=1)
