from polycraft_nov_det.data.cifar_loader import torch_cifar
from polycraft_nov_det.models.autonovel_resnet import AutoNovelResNet
from polycraft_nov_det.model_utils import load_autonovel_resnet
import polycraft_nov_det.train as train

mode = "self_supervised"

if mode == "self_supervised":
    # get dataloaders, note that batch is effectively 128 becuase of 4 rotations per image
    batch_size = 32
    _, _, (train_loader, _, test_loader) = torch_cifar(
        range(5), batch_size, include_novel=True, rot_loader="rotnet")
    # get model instance
    model = AutoNovelResNet(4, 0)
    # start model training
    model_label = train.model_label(model, range(4))
    train.train_self_supervised(model, model_label, train_loader, test_loader, gpu=1)
elif mode == "supervised":
    # get dataloaders
    batch_size = 128
    norm_targets, _, (train_loader, _, test_loader) = torch_cifar(range(5), batch_size)
    # get model instance
    model = load_autonovel_resnet("models/CIFAR10/self_supervised/200.pt", len(norm_targets), 0,
                                  reset_head=True, strict=False)
    model.freeze_layers()
    # start model training
    model_label = train.model_label(model, norm_targets)
    train.train_supervised(model, model_label, train_loader, test_loader, gpu=1)
elif mode == "autonovel":
    # get dataloaders
    # TODO test this should be 64 and not 128 due to transform twice
    batch_size = 64
    norm_targets, novel_targets, (train_loader, _, test_loader) = torch_cifar(
        range(5), batch_size, include_novel=True, rot_loader="consistent")
    # get model instance
    model = load_autonovel_resnet(
        "models/CIFAR10/supervised/100.pt", len(norm_targets), len(novel_targets), reset_head=False,
        strict=False, to_incremental=True)
    # start model training
    model_label = train.model_label(model, norm_targets)
    train.train_autonovel(model, model_label, train_loader, test_loader, norm_targets, gpu=1)
