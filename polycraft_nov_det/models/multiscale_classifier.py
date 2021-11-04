import torch
import torch.nn as nn


class MultiscaleClassifierConv(nn.Module):
    """
    Binary classifier model trained on the one-dimensional (avgd. over all 
    colour channels) rec.-error array of each scale
    """
    def __init__(self, conv_output_len):
        super(MultiscaleClassifierConv, self).__init__()
        self.conv_0_5 = nn.Conv2d(1, 1, 3)
        self.conv_0_75 = nn.Conv2d(1, 1, 3)
        self.conv_1 = nn.Conv2d(1, 1, 3)

        self.fc1 = nn.Linear(conv_output_len, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        input_0_5, input_0_75, input_1 = inputs
        out_0_5 = torch.flatten(self.conv_0_5(input_0_5), start_dim=1)
        out_0_75 = torch.flatten(self.conv_0_75(input_0_75), start_dim=1)
        out_1 = torch.flatten(self.conv_1(input_1), start_dim=1)
        out = self.relu(torch.cat((out_0_5, out_0_75, out_1), dim=1))

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc_out(out))

        return out


class MultiscaleClassifierFeatureVector(nn.Module):
    """
    Binary classifier model trained on a feature vectors. The feature vectors
    are either composed of the maximum rec.-error at each scale or the 
    maximum rec.- error (at each scale) and the distances between their 
    "patch-positions" (between each scale).
    """
    def __init__(self, ipt_channel):
        super(MultiscaleClassifierFeatureVector, self).__init__()
        self.ll1 = nn.Linear(ipt_channel, 12)
        self.ll2 = nn.Linear(12, 12)
        self.llout = nn.Linear(12, 1)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        linear1 = self.relu(self.ll1(inputs))
        linear2 = self.relu(self.ll2(linear1))
        output = self.sigmoid(self.llout(linear2))

        return output


class MultiscaleClassifierConvFeatComp3Models(nn.Module):
    """
    Binary classifier model trained on the three-dimensional rec.-error array 
    of each scale computed based on autoencoder models with patches 3x32x32. 
    The rec.-errors of the individual colour channels are not
    averaged for this model. 
    """
    def __init__(self, nc, conv_output_len):
        super(MultiscaleClassifierConvFeatComp3Models, self).__init__()
        self.conv_0_5 = nn.Conv2d(nc, 1, 3)
        self.conv_0_75 = nn.Conv2d(nc, 1, 3)
        self.conv_1 = nn.Conv2d(nc, 1, 3)

        self.fc1 = nn.Linear(conv_output_len, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 9)

        self.ll1 = nn.Linear(9, 12)
        self.ll2 = nn.Linear(12, 12)
        self.llout = nn.Linear(12, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        input_0_5, input_0_75, input_1 = inputs
        out_0_5 = torch.flatten(self.conv_0_5(input_0_5), start_dim=1)
        out_0_75 = torch.flatten(self.conv_0_75(input_0_75), start_dim=1)
        out_1 = torch.flatten(self.conv_1(input_1), start_dim=1)
        out = self.relu(torch.cat((out_0_5, out_0_75, out_1), dim=1))

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc_out(out))

        out = self.relu(self.ll1(out))
        out = self.relu(self.ll2(out))
        output = self.sigmoid(self.llout(out))

        return output
    
    
class MultiscaleClassifierConvFeatComp4Models(nn.Module):
    """
    Binary classifier model trained on the three-dimensional rec.-error array 
    of each scale. At scale 0.5 and scale 0.75 we have one model each with
    patches of size 3x32x32, at scale 1 we have one model based on patch size
    3x32x32 and an additional model based on patch size 3x16x16.
    The rec.-errors of the individual colour channels are not
    averaged for this model. 
    """
    def __init__(self, nc, conv_output_len):
        super(MultiscaleClassifierConvFeatComp4Models, self).__init__()
        self.conv_0_5 = nn.Conv2d(nc, 1, 3)
        self.conv_0_75 = nn.Conv2d(nc, 1, 3)
        self.conv_1 = nn.Conv2d(nc, 1, 3)
        self.conv_1b = nn.Conv2d(nc, 1, 3)

        self.fc1 = nn.Linear(conv_output_len, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 9)

        self.ll1 = nn.Linear(9, 12)
        self.ll2 = nn.Linear(12, 12)
        self.llout = nn.Linear(12, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        input_0_5, input_0_75, input_1, input_1b = inputs
        out_0_5 = torch.flatten(self.conv_0_5(input_0_5), start_dim=1)
        out_0_75 = torch.flatten(self.conv_0_75(input_0_75), start_dim=1)
        out_1 = torch.flatten(self.conv_1(input_1), start_dim=1)
        out_1b = torch.flatten(self.conv_1b(input_1b), start_dim=1)
        out = self.relu(torch.cat((out_0_5, out_0_75, out_1, out_1b), dim=1))

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc_out(out))

        out = self.relu(self.ll1(out))
        out = self.relu(self.ll2(out))
        output = self.sigmoid(self.llout(out))

        return output
