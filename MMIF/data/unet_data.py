import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UNet_Dataset(Dataset):
    def __init__(self, data_path, mode=None, transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform


        self.images = os.path.join()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
