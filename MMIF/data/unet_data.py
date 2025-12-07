from pathlib import Path

import torch
from numpy.ma.core import masked_greater_equal
from torch.utils.data import Dataset
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse

class UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mask_ratio=0.3):
        self.data_path = Path(data_path)
        self.images = sorted(self.data_path.glob("*"))
        self.transform = transform
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        gt = img.clone()
        mask = torch.rand_like(img)
        masked_img = img * mask

        return masked_img, gt


class DWT_UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mode='low', device='gpu', mask_ratio=0.3):
        self.data_path = Path(data_path)
        self.images = sorted(self.data_path.glob("*"))
        self.transform = transform
        self.mode = mode
        self.device = device
        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        img = img.to(self.device)
        LL, High_list = self.dwt(img.unsqueeze(0))

        High = torch.cat(High_list, dim=1)

        LL = LL.squeeze(0)
        High = High.squeeze(0)

        mask_ll = (torch.rand_like(LL) > self.mask_ratio).float()
        mask_high = (torch.rand_like(High) > self.mask_ratio).float()

        masked_LL = LL * mask_ll
        masked_High = High * mask_high


        if self.mode == 'low':
            return masked_LL, LL
        else:
            return masked_High, High



