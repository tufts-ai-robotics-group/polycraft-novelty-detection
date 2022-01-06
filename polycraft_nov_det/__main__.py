from polycraft_nov_det.data.cifar_loader import torch_cifar
from polycraft_nov_det.models.disc_resnet import DiscResNet
import polycraft_nov_det.train as train


# get dataloaders
batch_size = 128
_, _, (train_loader, valid_loader, _) = torch_cifar(batch_size, include_novel=True, rot_loader=True)
# get model instance
model = DiscResNet(4, 0)
# start model training
model_label = train.model_label(model, range(4))
train.train_self_supervised(model, model_label, train_loader, valid_loader, gpu=1)
