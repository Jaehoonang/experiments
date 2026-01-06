import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
from pathlib import Path

from models.CViT import CViTFlow
from data.fusiontransformer_data import Fusiondataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\infrared")
dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\visible")

dataset = Fusiondataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = CViTFlow(in_channels=1, embed_dim=768, patch_size=8, num_heads=12, stride=7).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 500
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