import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
from pathlib import Path

from models.fusion_transformer import ViTFlow
from data.fusiontransformer_data import Fusiondataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\infrared")
dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\visible")

dataset = Fusiondataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = ViTFlow(in_channels=1, img_size=224, emb_size=768, patch_size=8, num_heads=12, out_channels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def representation_loss(m1_out, m2_out,m1_in, m2_in, w_self=1.0, w_cross=0.1):
    # Self-consistency (identity preservation)
    loss_self_1 = F.l1_loss(m1_out, m1_in.detach())
    loss_self_2 = F.l1_loss(m2_out, m2_in.detach())
    loss_self = loss_self_1 + loss_self_2

    # Weak cross alignment (정보 교환 유도, collapse 방지)
    # cos = F.cosine_similarity(m1_out, m2_out, dim=-1)
    # loss_cross = 1 - cos.mean()
    # + w_cross * loss_cross

    return w_self * loss_self

def gradient_loss(output, target):
    dx_o = output[:, :, :, 1:] - output[:, :, :, :-1]
    dy_o = output[:, :, 1:, :] - output[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_o, dx_t) + F.l1_loss(dy_o, dy_t)

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    # img1, img2: [B, C, H, W]
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    sigma1 = F.avg_pool2d(img1 * img1, window_size, 1, window_size//2) - mu1**2
    sigma2 = F.avg_pool2d(img2 * img2, window_size, 1, window_size//2) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))

    return ssim_map.mean()

epochs = 50
save_dir = "ViT_checkpoints"
os.makedirs(save_dir, exist_ok=True)

best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    epoch_bar = tqdm(dataloader,desc=f"[Epoch {epoch+1}/{epochs}]",leave=False)

    for modal1_img, modal2_img in epoch_bar:
        modal1_img, modal2_img  = modal1_img.to(device), modal2_img.to(device)
        # print(modal1_img, modal2_img)
        optimizer.zero_grad()

        with torch.no_grad():
            modal1_in = model.embedding1(modal1_img)
            modal2_in = model.embedding2(modal2_img)

        modal1_out, modal2_out = model(modal1_img, modal2_img)
        recon_loss = F.l1_loss(modal1_out, modal1_img) + F.l1_loss(modal2_out, modal2_img)
        # edge_loss = gradient_loss(modal1_out, modal1_img) + gradient_loss(modal2_out, modal2_img)
        # ssim_loss = 1 - ssim(modal1_out, modal1_img) + 1 - ssim(modal2_out, modal2_img)

        # re_loss = representation_loss(modal1_out, modal2_out, modal1_img, modal2_img)
        # deco_loss = decoder_loss(modal1_out, modal2_out, modal1_img, modal2_img)

        loss = recon_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}", repr=f"{recon_loss.item():.4f}")
        # fuse = f"{fu_loss.item():.4f}"
    avg_loss = epoch_loss / len(dataloader)

    tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Decoder Loss: {recon_loss:.4f} | Representation Loss: {recon_loss:.4f} | Avg Loss: {avg_loss:.4f}")

    #| FusionLoss: {fu_loss: .4f}

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": best_loss
        }, os.path.join(save_dir, "best_representation_model.pth"))

        tqdm.write(f"Best model saved at {epoch + 1} epoch!")