from torch.utils.data import Dataset
from PIL import Image


class VMAE_Dataset(Dataset):
    def __init__(self, modal1_dir, modal2_dir, transform=None):
        self.modal1_dir = modal1_dir
        self.modal2_dir = modal2_dir

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

        return modal1_img, modal2_img