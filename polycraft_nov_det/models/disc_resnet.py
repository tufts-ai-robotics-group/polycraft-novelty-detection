import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class DiscResNet(resnet.ResNet):
    def __init__(self, num_labeled_classes, num_unlabeled_classes):
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        self.num_classes = num_labeled_classes + num_unlabeled_classes
        # init according to resnet18()
        super().__init__(
            resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_labeled_classes)
        # add additional fully connected head for unlabeled
        self.fc_unlabeled = nn.Linear(512*resnet.BasicBlock.expansion, num_unlabeled_classes)

    def init_incremental(self):
        # copy current labeled head weights
        save_weight = self.fc.weight.data.clone()
        save_bias = self.fc.bias.data.clone()
        # expand labeled head to include unlabeled class outputs
        self.fc = nn.Linear(512*resnet.BasicBlock.expansion, self.num_classes)
        # load labeled head weights, initializing unlabeled bias to just below the min
        self.fc.weight.data[:self.num_labeled_classes] = save_weight
        self.fc.bias.data[:] = torch.min(save_bias) - 1
        self.fc.bias.data[:self.num_labeled_classes] = save_bias

    def freeze_layers(self):
        for name, param in self.named_parameters():
            if 'fc' not in name and 'layer4' not in name:
                param.requires_grad = False

    def forward(self, x):
        # based on parent's implementation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # save fc1 features
        feat = x
        label_pred = self.fc(x)
        unlabel_pred = self.fc_unlabeled(x)
        return label_pred, unlabel_pred, feat
