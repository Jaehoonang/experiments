from models.UNet import UNet
from data.unet_data import UNet_Dataset, DWT_UNet_Dataset

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'current device : {device}')
# data_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest"
#
#
# image_size = 512
# transform = transforms.Compose([
#     transforms.Grayscale(),
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
#
# train_dataset = UNet_Dataset(data_path + '/train', transform=transform)
# test_dataset = UNet_Dataset(data_path + '/test', transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True )
# test_loader = DataLoader(test_dataset, shuffle=False)
#
# criterion = nn.L1Loss()
#
# model = UNet(in_channels=1, out_channels=1).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)
#
# os.makedirs("checkpoints", exist_ok=True)
# num_epochs = 5
# best_loss = float('inf')
#
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#
#     with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
#         for masked, gt in pbar:
#             masked, gt = masked.to(device), gt.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(masked)
#             loss = criterion(outputs, gt)
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#             pbar.set_postfix(loss=train_loss / (pbar.n + 1))
#
#     print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")
#
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for masked, gt in test_loader:
#             masked = masked.to(device)
#             gt = gt.to(device)
#             outputs = model(masked)
#             loss = criterion(outputs, gt)
#             test_loss += loss.item()
#
#     avg_test_loss = test_loss / len(test_loader)
#     print(f"[Test] Loss: {avg_test_loss:.4f}")
#
#     checkpoint = {
#         'epoch': epoch + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': train_loss / len(train_loader),
#         'test': test_loss
#     }
#     torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
#
#     if avg_test_loss < best_loss:
#         best_loss = avg_test_loss
#         torch.save(model.state_dict(), "checkpoints/best_model.pth")
#         print(f" Best model saved at epoch {epoch + 1} ")
#
#         print("epoch finished!")
#
# print("training finished!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'current device : {device}')
data_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest"


image_size = 512
transform = transforms.Compose([
    transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

train_dataset = DWT_UNet_Dataset(data_path + '/train', mode='low', transform=transform, device=device)
test_dataset = DWT_UNet_Dataset(data_path + '/test', mode='low', transform=transform, device=device)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=False)

criterion = nn.L1Loss()

model = UNet(in_channels=1, out_channels=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

os.makedirs("checkpoints", exist_ok=True)
num_epochs = 5
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for masked, gt in pbar:
            masked, gt = masked.to(device), gt.to(device)

            optimizer.zero_grad()
            outputs = model(masked)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / (pbar.n + 1))

    print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for masked, gt in test_loader:
            masked = masked.to(device)
            gt = gt.to(device)
            outputs = model(masked)
            loss = criterion(outputs, gt)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Test] Loss: {avg_test_loss:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss / len(train_loader),
        'test': test_loss
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f" Best model saved at epoch {epoch + 1} ")

        print("epoch finished!")

print("training finished!")

