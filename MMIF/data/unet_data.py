from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


# image_size = 512
# transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])


class UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.images = sorted(self.data_path.glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img
