import os
from PIL import Image
from torchvision import transforms
from pytorch_wavelets import DWTForward, DWTInverse

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
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

    X = trans(img)
    X = X.unsqueeze(0)
    # print(X.shape)
    return X.to(device)

def ex_data_dwt(root_dir):
    img = Image.open(root_dir).convert("L")

    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    X = trans(img).to(device)
    X = X.unsqueeze(0)
    xfm = DWTForward(J=1, mode="periodization", wave='haar').to(device)
    x_low, x_high = xfm(X)
    x_low = x_low.to(device)

    if isinstance(x_high, (list, tuple)):
        x_high = [h.to(device) for h in x_high]
    else:
        x_high = x_high.to(device)

    return x_low, x_high

def ex_data3(root_dir):
    img = Image.open(root_dir).convert("YCbCr")
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

    X = trans(img)
    X = X.unsqueeze(0)
    return X.to(device)

def ex_data4(root_dir):
    img = Image.open(root_dir).convert("RGB")
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

    X = trans(img)
    X = X.unsqueeze(0)
    return X.to(device)