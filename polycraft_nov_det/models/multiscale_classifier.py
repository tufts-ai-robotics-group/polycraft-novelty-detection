import torch
import torch.nn as nn


class MultiscaleClassifierConv(nn.Module):
    def __init__(self):
        super(MultiscaleClassifierConv, self).__init__()
        self.conv_0_5 = nn.Conv2d(1, 1, 3)
        self.conv_0_75 = nn.Conv2d(1, 1, 3)
        self.conv_1 = nn.Conv2d(1, 1, 3)

        conv_output_len = 100
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
    