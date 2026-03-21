from torch.utils.data import Dataset
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse


class VMAE_Dataset(Dataset):
    def __init__(self, modal1_dir, modal2_dir, transform=None, freq='low', device='gpu'):
        self.modal1_dir = modal1_dir
        self.modal2_dir = modal2_dir

        self.modal1_images = sorted(self.modal1_dir.glob("*"))
        self.modal2_images = sorted(self.modal2_dir.glob("*"))
        self.transform = transform

        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(device)
        self.freq = freq

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

        if self.freq == 'low':
             dal1_low, _ = self.dwt(modal1_img)
             dal2_low, _ = self.dwt(modal2_img)

             return dal1_low, dal2_low

        else:
            _, dal1_high = self.dwt(modal1_img)
            _, dal2_high = self.dwt(modal2_img)

            return dal1_high, dal2_high


