import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, code_length, class_num):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(code_length, code_length),
            nn.LeakyReLU(),
            nn.Linear(code_length, code_length),
            nn.LeakyReLU(),
        )
        self.output = nn.Sequential(nn.Linear(code_length, class_num), nn.LogSoftmax(dim=1))

    def forward(self, x):
        h = self.hidden(x)
        o = self.output(h)
        return o
