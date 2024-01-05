import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import v2
from skimage import io
import os


class RobotImageDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, multi_input=False, file_names=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[['ImageID','Joint3','Joint4','Joint5', 'Joint6']]
        self.root_dir = root_dir
        self.multi_input = multi_input
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
        if self.multi_input:
            self.transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32),
                    v2.ToTensor(),
                    v2.Resize(size=(224, 224), antialias=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], #default resnet norm
                                std=[0.229, 0.224, 0.225]),
                    v2.ToTensor()
                ])
        else:
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
        out = [im for im in images] if self.multi_input else torch.cat([im for im in images],dim = 0)
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