import os
from PIL import Image
from torchvision import transforms

import torch
import torch.utils
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ex_data1(root_dir):
    img = Image.open(root_dir).convert("L")
    #
    trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor()])

    X = trans(img).unsqueeze(0)
    # print(X.shape)
    return X.to(device)

def ex_data2(root_dir):
    img = Image.open(root_dir).convert("L")
    #
    trans = transform = transforms.Compose([
        transforms.Grayscale(),            # 1채널로 변환
        transforms.Resize((224, 224)),     # 크기 맞춤
        transforms.ToTensor(),             # [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5]) # [-1, 1]로 깔끔하게 매핑
])

    X = trans(img)
    X = X.unsqueeze(0)
    # print(X.shape)
    return X.to(device)