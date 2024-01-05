import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision.transforms import v2
from helper import RobotNet
from helper import RobotImageDataset
import io
import os
import matplotlib.pyplot as plt

multi_model = RobotNet(128,32,n_input_channels=3)
multi_model.load_state_dict(torch.load('model_state_dict_epoch14_3view.pth',map_location=torch.device('cpu')))

robotdata = RobotImageDataset('UR5_positions_3.csv', './UR5_images_3', file_names=['side_view_','front_view_','top_view_'])
loader = DataLoader(robotdata, batch_size=10, shuffle=True)

multi_model.eval()

sample = next(iter(loader))

preds = multi_model(sample['images'])
gt = sample['joint_values']

print(preds)
print(gt)

def get_prediction(img):
    # multi_model.eval()
    tensor = RobotImageDataset.process_images(robotdata,img)
    # plt.imshow(tensor.permute(1, 2, 0))
    tensor = tensor.reshape((1,3,224,224))
    
    pred = multi_model(tensor).detach().numpy()

    

    return pred
