import torch.nn as nn
from torchvision.models import resnet18
import torch
from torchvision.models.resnet import ResNet18_Weights


class RobotNet(nn.Module):
    def __init__(self, n_hidden_units1, n_hidden_units2, n_inputs=1, freeze_resnet = False):
        super().__init__()
        self.num_inputs = n_inputs
        self.in_dim1 = n_hidden_units1
        self.in_dim2 = n_hidden_units2
        self.weights = ResNet18_Weights.DEFAULT

        self.base_model = resnet18(weights=self.weights)
        self.base_layers = list(self.base_model.children())
        self.flatten = nn.Flatten()

        # freeze resnet layers
        if freeze_resnet:
            for idx,child in enumerate(self.base_layers):
              for param in child.parameters():
                param.requires_grad = False


        self.resnet_layers = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            *self.base_layers[1:-1])

        self.linear_output = nn.Sequential(
            nn.Linear(in_features=512*self.num_inputs, out_features=self.in_dim1),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim1, out_features=self.in_dim2),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim2, out_features=4)
        )


    def forward(self, inputs):

        if self.num_inputs > 1:
            for idx in range(len(inputs)):
                inputs[idx] = self.resnet_layers(inputs[idx])
            out = self.linear_output(torch.cat(inputs,dim=1).view(-1,512*len(inputs)))
        else:
            out = self.resnet_layers(inputs)
            out = self.linear_output(self.flatten(out))

        return out