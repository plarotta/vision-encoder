import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
from skimage import io
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class RobotImageDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, file_names=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[['ImageID','Joint3','Joint4','Joint5', 'Joint6']]
        self.root_dir = root_dir
        if not file_names:
            self.file_names = [
                              "side_view_",
                              "front_view_",
                              "top_view_",
                              "corner1_view_",
                              "corner2_view_",
                              "side_depth_view_",
                              "front_depth_view_",
                              "top_depth_view_",
                              "corner1_depth_view_",
                              "corner2_depth_view_"
                              ]
        else:
            self.file_names = file_names

        #TODO: improve transform so that the final image view is more zoomed in
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32),
                v2.ToTensor(),
                v2.Resize(size=(224, 224), antialias=True),
                v2.Grayscale(),
                v2.ToTensor()
            ])

    def process_images(self, images):
        for im_idx in range(len(images)):
            if len(images[im_idx].shape) > 2:
                images[im_idx] = images[im_idx][:,:,:3]
            images[im_idx] = self.transform(images[im_idx])
        out = torch.cat([im for im in images],dim = 0)
        return(out)

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = [io.imread(os.path.join(self.root_dir, f + str(idx) + '.png')) for f in self.file_names]
        im_data = self.process_images(images)
        joint_values = self.annotations.iloc[idx, 1:].to_numpy(dtype=float)
        sample = {'images': im_data, 'joint_values': joint_values}
        return sample




class RobotNet(nn.Module):
    def __init__(self, n_hidden_units1, n_hidden_units2, n_input_channels, num_inputs=1, multi_input=False, freeze_resnet=False):
        super().__init__()
        self.multi_input = multi_input
        self.in_dim1 = n_hidden_units1
        self.in_dim2 = n_hidden_units2
        self.weights = ResNet18_Weights.DEFAULT

        self.base_model = resnet18(weights=self.weights)
        self.base_layers = list(self.base_model.children())
        self.flatten = nn.Flatten()
        self.num_inputs = num_inputs

        # freeze resnet layers
        if freeze_resnet:
            for idx,child in enumerate(self.base_layers):
              for param in child.parameters():
                param.requires_grad = False


        self.resnet_layers = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            *self.base_layers[1:-1])

        self.linear_output = nn.Sequential(
            nn.Linear(in_features=512*self.num_inputs, out_features=self.in_dim1),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim1, out_features=self.in_dim2),
            nn.ReLU(),
            nn.Linear(in_features=self.in_dim2, out_features=4)
        )

    def forward(self, inputs):

        if self.multi_input:
            inputs = inputs.transpose(0,1)
            temp = [0 for _ in range(inputs.shape[0])]
            for idx in range(len(inputs)):
                temp[idx] = inputs[idx].unsqueeze(1)
                temp[idx] = self.resnet_layers(temp[idx])
            out = self.linear_output(torch.cat(temp,dim=1).view(-1,512*len(inputs)))
            del temp
        else:
            out = self.resnet_layers(inputs)
            out = self.linear_output(self.flatten(out))

        return out


