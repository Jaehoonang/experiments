import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
from pathlib import Path

from torchvision.models import VGG19_Weights, vgg19

from data.VMAE_data import VMAE_Dataset
from models.VMAE import VMAE
from utils.loss import GradientLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 device {device}")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\infrared")
dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\visible")

dataset= VMAE_Dataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = VMAE(in_channels=1, patch_size=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:14].to(device)
vgg.eval()
def to_3ch(x):
    return x.repeat(1, 3, 1, 1)

mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
std  = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

def vgg_norm(x):
    return (x - mean) / std

beta = 1e-4
epochs = 2000
save_dir = "VMAE_checkpoints"
os.makedirs(save_dir, exist_ok=True)
grad_loss_fn = GradientLoss()
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    epoch_bar = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]", leave=False)

    for modal1_img, modal2_img in epoch_bar:
        modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)

        optimizer.zero_grad()

        output, mask, posterior = model(modal1_img, modal2_img)
        target = torch.max(modal1_img, modal2_img)

        # l2 loss with summation reconstruction
        # recon_loss = ((output - target) ** 2).mean(dim=(1, 2, 3))

        # 가장 최신 recon_loss
        #######################################
        recon_loss = F.l1_loss(output, target, reduction='none')
        recon = recon_loss.mean(dim=(1, 2, 3))  # [B]
        focus_weight = mask.float().mean(dim=1)
        recon_loss = (recon * (1 + focus_weight)).mean()
        #######################################

        # gpt recons loss
        patch_mask = mask  #
        B, _, H, W = output.shape
        p = model.module.Patch_Posi.patch_size
        h = H // p
        w = W // p
        img_mask = patch_mask.reshape(B, h, w)
        img_mask = img_mask.repeat_interleave(p, 1).repeat_interleave(p, 2)
        img_mask = img_mask.unsqueeze(1)  # [B,1,H,W]
        visible_mask = 1 - img_mask

        recon = F.l1_loss(output, target, reduction='none')
        recon_loss1 = (recon * img_mask).sum() / (img_mask.sum() + 1e-6)
        recon_loss2 = (recon * visible_mask).sum() / (visible_mask.sum() + 1e-6)
        # gpt recons loss

        # recon_loss = (output - max(modal1_img, modal2_img)).mean(dim=(1, 2, 3))
        # focus_weight = mask.float().mean(dim=1)
        # recon_loss = (recon_loss * (1 + focus_weight)).mean()

        with torch.no_grad():
            target_feat = vgg(vgg_norm(to_3ch(torch.max(modal1_img, modal2_img))))

        output_feat = vgg(vgg_norm(to_3ch(output)))
        perceptual_loss = F.l1_loss(output_feat, target_feat)

        grad_loss = grad_loss_fn(output, modal1_img, modal2_img)

        # KL loss
        mu = posterior.mean
        logvar = posterior.logvar

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss1 + recon_loss2 + perceptual_loss + beta * kl_loss + grad_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}", recon1=f"{recon_loss1.item():.4f}",
                              recon2=f"{recon_loss2.item():.4f}", percepual=f"{perceptual_loss.item():.4f}",
                              KL=f"{kl_loss.item():.4f}", grad=f"{grad_loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)

    tqdm.write(
        f"Epoch [{epoch + 1}/{epochs}] | Reconstruction Loss1: {recon_loss1:.4f} | Reconstruction Loss2: {recon_loss2:.4f} |Perceptual Loss: {perceptual_loss:.4f} | KL Loss: {kl_loss:.4f} | grad Loss: {grad_loss:.4f} | Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": best_loss
        }, os.path.join(save_dir, f"best_representation_model.pth"))

        tqdm.write(f"Best model saved at {epoch + 1} epoch!")

