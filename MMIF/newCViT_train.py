import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
from pathlib import Path
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from models.newCViT import newCViTFlow
from data.fusiontransformer_data import Fusiondataset
from utils.loss import GradientLoss, IntensityLoss

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

model = newCViTFlow(in_channels=1, embed_dim=64, patch_size=7, num_heads=8, dim=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
grad_loss_fn = GradientLoss()
intensity_loss_fn = IntensityLoss()

epochs = 300
save_dir = "newCVT_checkpoints"
os.makedirs(save_dir, exist_ok=True)

best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    epoch_bar = tqdm(dataloader,desc=f"[Epoch {epoch+1}/{epochs}]",leave=False)

    for modal1_img, modal2_img in epoch_bar:
        modal1_img, modal2_img  = modal1_img.to(device), modal2_img.to(device)
        optimizer.zero_grad()


        fused_out = model(modal1_img, modal2_img)
        intensity_loss = intensity_loss_fn(fused_out, modal1_img, modal2_img)
        grad_loss = grad_loss_fn(fused_out, modal1_img, modal2_img)

        loss = intensity_loss + grad_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{intensity_loss.item():.4f}", grad=f"{grad_loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)

    tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Recon Loss: {intensity_loss:.4f} | Grad Loss: {grad_loss:.4f} | Avg Loss: {avg_loss:.4f}")


    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": best_loss
        }, os.path.join(save_dir, "best_representation_model.pth"))

        tqdm.write(f"Best model saved at {epoch + 1} epoch!")