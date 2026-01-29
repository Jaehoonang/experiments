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

model = VMAE(in_channels=1).to(device)
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
epochs = 1000
save_dir = "VMAE_checkpoints"
os.makedirs(save_dir, exist_ok=True)

best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    epoch_bar = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]", leave=False)

    for modal1_img, modal2_img in epoch_bar:
        modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)

        optimizer.zero_grad()

        output, mask, posterior = model(modal1_img, modal2_img)
        target = modal1_img + modal2_img

        recon_loss = ((output - target) ** 2).mean(dim=(1, 2, 3))
        focus_weight = mask.float().mean(dim=1)
        recon_loss = (recon_loss * (1 + focus_weight)).mean()

        with torch.no_grad():
            target_feat = vgg(vgg_norm(to_3ch(modal1_img + modal2_img)))

        output_feat = vgg(vgg_norm(to_3ch(output)))
        perceptual_loss = F.l1_loss(output_feat, target_feat)

        # KL loss
        mu = posterior.mean
        logvar = posterior.logvar

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + perceptual_loss + beta*kl_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", percepual=f"{perceptual_loss.item():.4f}", KL=f"{kl_loss.item():.4f}" )

    avg_loss = epoch_loss / len(dataloader)

    tqdm.write(f"Epoch [{epoch + 1}/{epochs}] | Reconstruction Loss: {recon_loss:.4f} | Perceptual Loss: {perceptual_loss:.4f} | KL Loss: {kl_loss:.4f} | Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": best_loss
        }, os.path.join(save_dir, f"best_representation_model.pth"))

        tqdm.write(f"Best model saved at {epoch + 1} epoch!")

