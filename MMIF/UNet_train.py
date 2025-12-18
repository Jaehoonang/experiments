from models.UNet import UNet
from data.unet_data import UNet_Dataset, DWT_UNet_Dataset, MMDWT_UNet_Dataset, ex1_MMDWT_UNet_Dataset, MMDWT_UNet_Bottom_Dataset, weight_MMDWT_UNet_Dataset

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
# train_dataset = DWT_UNet_Dataset(data_path + '/train', mode='low', transform=transform, device=device)
# test_dataset = DWT_UNet_Dataset(data_path + '/test', mode='low', transform=transform, device=device)
#
# train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
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

def train_unet(in_channels, out_channels, mode, data_path, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device : {device}')

    image_size = 512
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = DWT_UNet_Dataset(data_path+ '/train', mode=mode, transform=transform, device=device)
    test_dataset = DWT_UNet_Dataset(data_path+ '/test', mode=mode, transform=transform, device=device)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    criterion = nn.L1Loss()

    model = UNet(in_channels, out_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{mode}", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (masked, gt) in enumerate(pbar):
                masked, gt = masked.to(device), gt.to(device)

                optimizer.zero_grad()
                outputs = model(masked)
                loss = criterion(outputs, gt)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=train_loss / (i + 1))

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
            'test_loss': avg_test_loss
        }
        torch.save(checkpoint, f"checkpoints/{mode}/checkpoint_epoch_{epoch + 1}.pth")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), f"checkpoints/{mode}/best_model.pth")
            print(f" Best {mode} model saved at epoch {epoch + 1} ")
            print(f" {epoch + 1} finished!")

    print("training finished!")

def train_MMunet(in_channels, out_channels, mode, data_path, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device : {device}')

    image_size = 512
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MMDWT_UNet_Dataset(data_path+ '/train', mode=mode, transform=transform, device=device)
    test_dataset = MMDWT_UNet_Dataset(data_path+ '/test', mode=mode, transform=transform, device=device)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    criterion = nn.L1Loss()

    model = UNet(in_channels, out_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{mode}", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (inputs, mask, gt) in enumerate(pbar):
                inputs, mask, gt = inputs.to(device), mask.to(device), gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                optimizer.zero_grad()
                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=train_loss / (i + 1))

        print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, mask, gt in test_loader:

                inputs = inputs.to(device)
                mask = mask.to(device)
                gt = gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"[Test] Loss: {avg_test_loss:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / len(train_loader),
            'test_loss': avg_test_loss
        }
        torch.save(checkpoint, f"checkpoints/{mode}/checkpoint_epoch_{epoch + 1}.pth")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), f"checkpoints/{mode}/best_model.pth")
            print(f" Best {mode} model saved at epoch {epoch + 1} ")
            print(f" {epoch + 1} finished!")

    print("training finished!")

def train_bottom_MMunet(in_channels, out_channels, mode, data_path, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device : {device}')

    image_size = 512
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MMDWT_UNet_Bottom_Dataset(data_path+ '/train', mode=mode, transform=transform, device=device)
    test_dataset = MMDWT_UNet_Bottom_Dataset(data_path+ '/test', mode=mode, transform=transform, device=device)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    criterion = nn.L1Loss(reduction='none')

    model = UNet(in_channels, out_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{mode}", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (inputs, mask, gt) in enumerate(pbar):
                inputs, mask, gt = inputs.to(device), mask.to(device), gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B*M, C, H, W)
                gt = gt.view(B*M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                optimizer.zero_grad()
                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=train_loss / (i + 1))

        print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, mask, gt in test_loader:

                inputs = inputs.to(device)
                mask = mask.to(device)
                gt = gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"[Test] Loss: {avg_test_loss:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / len(train_loader),
            'test_loss': avg_test_loss
        }
        torch.save(checkpoint, f"checkpoints/{mode}/checkpoint_epoch_{epoch + 1}.pth")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), f"checkpoints/{mode}/best_model.pth")
            print(f" Best {mode} model saved at epoch {epoch + 1} ")
            print(f" {epoch + 1} finished!")

    print("training finished!")

def train_ano_MMunet(in_channels, out_channels, mode, data_path, num_epochs, annotation_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device : {device}')

    image_size = 512
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ex1_MMDWT_UNet_Dataset(data_path+ '/train',annotation_path, mode=mode, transform=transform, device=device)
    test_dataset = ex1_MMDWT_UNet_Dataset(data_path+ '/test', annotation_path, mode=mode, transform=transform, device=device)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    criterion = nn.L1Loss()

    model = UNet(in_channels, out_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{mode}", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (inputs, mask, gt) in enumerate(pbar):
                inputs, mask, gt = inputs.to(device), mask.to(device), gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                optimizer.zero_grad()
                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=train_loss / (i + 1))

        print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, mask, gt in test_loader:

                inputs = inputs.to(device)
                mask = mask.to(device)
                gt = gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"[Test] Loss: {avg_test_loss:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / len(train_loader),
            'test_loss': avg_test_loss
        }
        torch.save(checkpoint, f"checkpoints/{mode}/checkpoint_epoch_{epoch + 1}.pth")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), f"checkpoints/{mode}/best_model.pth")
            print(f" Best {mode} model saved at epoch {epoch + 1} ")
            print(f" {epoch + 1} finished!")

    print("training finished!")

def train_wieght_ano_MMunet(in_channels, out_channels, mode, data_path, num_epochs, annotation_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device : {device}')

    image_size = 512
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = weight_MMDWT_UNet_Dataset(data_path+ '/train',annotation_path, mode=mode, transform=transform, device=device)
    test_dataset = weight_MMDWT_UNet_Dataset(data_path+ '/test', annotation_path, mode=mode, transform=transform, device=device)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    criterion = nn.L1Loss()

    model = UNet(in_channels, out_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{mode}", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (inputs, mask, gt) in enumerate(pbar):
                inputs, mask, gt = inputs.to(device), mask.to(device), gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                optimizer.zero_grad()
                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=train_loss / (i + 1))

        print(f"[Train] Epoch {epoch + 1} Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, mask, gt in test_loader:

                inputs = inputs.to(device)
                mask = mask.to(device)
                gt = gt.to(device)

                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                B, M, C, H, W = inputs.size()

                inputs = inputs.view(B * M, C, H, W)
                gt = gt.view(B * M, C, H, W)
                mask = mask.repeat_interleave(M, dim=0)
                mask = mask.expand(-1, C, -1, -1)

                outputs = model(inputs)

                pixel_loss = criterion(outputs, gt)
                loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-8)

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"[Test] Loss: {avg_test_loss:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / len(train_loader),
            'test_loss': avg_test_loss
        }
        torch.save(checkpoint, f"checkpoints/{mode}/checkpoint_epoch_{epoch + 1}.pth")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), f"checkpoints/{mode}/best_model.pth")
            print(f" Best {mode} model saved at epoch {epoch + 1} ")
            print(f" {epoch + 1} finished!")

    print("training finished!")


if __name__ == '__main__':
    low_mode = 'low'
    high_mode = 'high'

    num_epochs = 30
    data_path = r'C:\Users\12wkd\Desktop\experiments\MMIF\onlytest'
    annotation_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\annotation"
    low_in = 1
    low_out = 1
    # low_unet = train_unet(in_channels=low_in, out_channels=low_out, mode='low', num_epochs=10, data_path=data_path)

    high_in = 3
    high_out = 3
    # high_unet = train_unet(in_channels=high_in, out_channels=high_out, mode='high', num_epochs=10, data_path=data_path)

    # difference bottom 30 masking (conducted)
    train_bottom_MMunet(in_channels=high_in, out_channels=high_out, mode='high', num_epochs=num_epochs,data_path=data_path)
    train_bottom_MMunet(in_channels=low_in, out_channels=low_out, mode='low', num_epochs=num_epochs, data_path=data_path)

    # # difference top 30 masking (conducted)
    # train_MMunet(in_channels=low_in, out_channels=low_out, mode='low', num_epochs=num_epochs, data_path=data_path)
    # train_MMunet(in_channels=high_in, out_channels=high_out, mode='high', num_epochs=num_epochs, data_path=data_path)

    # # difference 30 + annotation
    # train_ano_MMunet(in_channels=low_in, out_channels=low_out, mode='low', num_epochs=num_epochs, data_path=data_path, annotation_path=annotation_path)
    # train_ano_MMunet(in_channels=high_in, out_channels=high_out, mode='high', num_epochs=num_epochs, data_path=data_path, annotation_path=annotation_path)

    # # (difference + annotation_weight) top 30
    # train_wieght_ano_MMunet(in_channels=high_in, out_channels=high_out, mode='high', num_epochs=num_epochs, data_path=data_path, annotation_path=annotation_path)
    # train_wieght_ano_MMunet(in_channels=low_in, out_channels=low_out, mode='low', num_epochs=num_epochs, data_path=data_path, annotation_path=annotation_path)
