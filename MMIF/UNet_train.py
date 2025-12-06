from models.UNet import UNet
from data.unet_data import UNet_Dataset

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest"


image_size = 512
transform = transforms.Compose([
    transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

train_dataset = UNet_Dataset(data_path + '/train', transform=transform)
test_dataset = UNet_Dataset(data_path + '/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True )
test_loader = DataLoader(test_dataset, shuffle=False)

criterion = nn.L1Loss()


model = UNet(in_channels=1, out_channels=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for imgs in pbar:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / (pbar.n + 1))

    print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")


model.eval()
test_loss = 0
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader):.4f}")

