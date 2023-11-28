import torch.nn as nn
from torchvision.models import resnet18
import torch


class RobotNet(nn.Module):
    def __init__(self, n_hidden_units1, n_hidden_units2, n_inputs=1):
        super().__init__()
        self.num_inputs = n_inputs
        self.in_dim1 = n_hidden_units1
        self.in_dim2 = n_hidden_units2

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # freeze resnet layers
        for idx,child in enumerate(self.base_layers):
          for param in child.parameters():
            param.requires_grad = False
        self.resnet_layers = nn.Sequential(
            *self.base_layers[:-1])

        self.linear_output = nn.Sequential(
            nn.Linear(in_features=512*self.num_inputs, out_features=self.in_dim1),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim1, out_features=self.in_dim2),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim2, out_features=5)
        )


    def forward(self, x1, x2=None, x3=None):
        # pseudo hard coded way of giving multi-input option
        x1 = self.resnet_layers(x1)
        if self.num_inputs == 2:
            x2 = self.resnet_layers(x2)
            out = self.linear_output(torch.cat([x1,x2],dim=1).view(-1,512*self.num_inputs))
        elif self.num_inputs == 3:
            x2 = self.resnet_layers(x2)
            x3 = self.resnet_layers(x3)
            out = self.linear_output(torch.cat([x1,x2,x3],dim=1).view(-1,512*self.num_inputs))
        else:
            out = x1

        return out

class VisionNet(nn.Module):
    def __init__(self, n_hidden_units1, n_hidden_units2, n_inputs=1):
        super().__init__()
        self.num_inputs = n_inputs
        self.in_dim1 = n_hidden_units1
        self.in_dim2 = n_hidden_units2

        self.base_model = vit_b_16(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # freeze resnet layers
        for idx,child in enumerate(self.base_layers):
          for param in child.parameters():
            param.requires_grad = False
        self.resnet_layers = nn.Sequential(
            *self.base_layers[:-1])

        self.linear_output = nn.Sequential(
            nn.Linear(in_features=512*self.num_inputs, out_features=self.in_dim1),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim1, out_features=self.in_dim2),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim2, out_features=5)
        )


    def forward(self, x1, x2=None, x3=None):
        # pseudo hard coded way of giving multi-input option
        x1 = self.resnet_layers(x1)
        if self.num_inputs == 2:
            x2 = self.resnet_layers(x2)
            out = self.linear_output(torch.cat([x1,x2],dim=1).view(-1,512*self.num_inputs))
        elif self.num_inputs == 3:
            x2 = self.resnet_layers(x2)
            x3 = self.resnet_layers(x3)
            out = self.linear_output(torch.cat([x1,x2,x3],dim=1).view(-1,512*self.num_inputs))
        else:
            out = x1

        return out
