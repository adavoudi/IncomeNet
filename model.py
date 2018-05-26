
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models

class IncomeNet(nn.Module):

    def __init__(self, pretrained=None):
        super(IncomeNet).__init__()

        self.base = models.resnet50(pretrained=pretrained==None)
        self.base.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.base.fc.in_features, 1, bias=True)
        )

        if pretrained:
            self.load_state_dict(torch.load(pretrained))

    def forward(self, inp):
        output = self.base(inp)
        return output
