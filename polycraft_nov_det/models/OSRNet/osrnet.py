import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class OSRNetCNN(resnet.ResNet):
    def __init__(self, num_classes, fc_dim=128):
        # init according to resnet34()
        super().__init__(
            resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        # update the fully connected layers
        self.fc1 = nn.Linear(self.fc.in_features, fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.fc.out_features)
        del self.fc

    def forward(self, x, return_fc1=False):
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
        # save fc1 features and return if requested
        x = self.fc1(x)
        fc1_features = x
        x = self.fc2(x)

        if return_fc1:
            return x, fc1_features
        return x


class OSRNetCS(nn.Module):
    def __init__(self, fc_dim=128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
