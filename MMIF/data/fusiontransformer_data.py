from torch.utils.data import Dataset
from pathlib import Path
import torch.nn as nn
import torch
from PIL import Image


class Fusiondataset(Dataset):
    def __init__(self, modal1_dir, modal2_dir, device='cuda', transform=None):
        self.modal1_dir = modal1_dir
        self.modal2_dir = modal2_dir
        self.device = device

        self.modal1_images = sorted(self.modal1_dir.glob("*"))
        self.modal2_images = sorted(self.modal2_dir.glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.modal1_images)

    def __getitem__(self, idx):
        modal1_path = self.modal1_images[idx]
        modal2_path = self.modal2_images[idx]

        modal1_img = Image.open(modal1_path).convert('L')
        modal2_img = Image.open(modal2_path).convert('L')

        if self.transform:
            modal1_img = self.transform(modal1_img)
            modal2_img = self.transform(modal2_img)

        modal1_img = modal1_img.to(self.device)
        modal2_img = modal2_img.to(self.device)

        return modal1_img, modal2_img
