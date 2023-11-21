import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import v2
from skimage import io
import os


class RobotImageDataset(Dataset):
    #TODO: make multi_input = False functional. Right now it fails 
    def __init__(self, csv_file: str, root_dir: str, multi_input = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[['ImageID','Joint1','Joint2','Joint3','Joint4','Joint5']]
        self.root_dir = root_dir
        self.multi_input = multi_input

        #TODO: improve transform so that the final image view is more zoomed in
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=(224, 224), antialias=True),
                v2.ToDtype(torch.float32),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], #default resnet norm
                             std=[0.229, 0.224, 0.225]),
                v2.ToTensor()
            ])

    def process_sample(self, top, side, front):
        # removing the empty channel
        top = top[:,:,:3]
        side = side[:,:,:3]
        front = front[:,:,:3]

        
        top_tens = self.transform(top)
        side_tens = self.transform(side)
        front_tens = self.transform(front)

        out = [side_tens, top_tens, front_tens] if self.multi_input else torch.cat([side_tens, top_tens, front_tens],dim = 0)

        return(out)

    def __len__(self):
        return(len(self.annotations))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        side_image_path = os.path.join(self.root_dir, 'side_view_'+ str(idx) + '.png')
        top_image_path = os.path.join(self.root_dir, 'top_view_'+str(idx) + '.png')
        front_image_path = os.path.join(self.root_dir, 'front_view_'+str(idx) + '.png')


        side_image = io.imread(side_image_path)
        top_image = io.imread(top_image_path)
        front_image = io.imread(front_image_path)

        im_data = self.process_sample(top_image, side_image, front_image, )

        joint_values = self.annotations.iloc[idx, 1:].to_numpy(dtype=float)

        sample = {'images': im_data, 'joint_values': joint_values}
        return sample