import os
from PIL import Image
from torchvision import transforms

import torch
import torch.utils
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ex_data(root_dir):
    img = Image.open(root_dir).convert("L")
    trans = transforms.Compose([transforms.Resize((512, 512)),
                            transforms.ToTensor()])

    X = trans(img).unsqueeze(0)
    print(X.shape)
    return X.to(device)