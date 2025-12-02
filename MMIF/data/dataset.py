import os
from PIL import Image
from torchvision import transforms

import torch
import torch.utils
from torch.utils.data import Dataset


trans = transforms.Compose([transforms.Resize((512, 512)),
                            transforms.ToTensor(),
                            transforms.Normalize(())])


class CustomDataset(Dataset):
    def __init__(self, root_dir, image_size = 512):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)
        self.image_size = image_size



    def __getitem__(self, index):


    def __len__(self):
        return len(self.image_paths)







if __name__ == "__main__":
