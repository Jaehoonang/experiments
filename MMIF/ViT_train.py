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
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dal1 = Path(r"C:/Users/12wkd/Desktop/experiments/MMIF/onlytest/train/visible")
dal2 = Path(r"C:/Users/12wkd/Desktop/experiments/MMIF/onlytest/train/infrared")

dataset = Fusiondataset(modal1_dir=dal1, modal2_dir=dal2, device=device, transform=transform)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = ViTFlow(in_channels=1, img_size=224, emb_size=768, patch_size=16, num_heads=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def representation_loss(
    m1_out, m2_out,
    m1_in, m2_in,
    w_self=1.0,
    w_cross=0.1
):

    # Self-consistency (identity preservation)
    loss_self_1 = F.l1_loss(m1_out, m1_in.detach())
    loss_self_2 = F.l1_loss(m2_out, m2_in.detach())
    loss_self = loss_self_1 + loss_self_2

    # Weak cross alignment (정보 교환 유도, collapse 방지)
    cos = F.cosine_similarity(m1_out, m2_out, dim=-1)
    loss_cross = 1 - cos.mean()

    return w_self * loss_self + w_cross * loss_cross

epochs = 50
save_dir = "ViT_checkpoints"
os.makedirs(save_dir, exist_ok=True)

best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # epoch tqdm
    epoch_bar = tqdm(
        dataloader,
        desc=f"[Epoch {epoch+1}/{epochs}]",
        leave=False
    )

    for modal1_img, modal2_img in epoch_bar:
        modal1_img = modal1_img.to(device)
        modal2_img = modal2_img.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            modal1_in = model.embedding1(modal1_img)
            modal2_in = model.embedding2(modal2_img)

        modal1_out, modal2_out = model(modal1_img, modal2_img)

        loss = representation_loss(
            modal1_out, modal2_out,
            modal1_in, modal2_in
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)

    tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f}")


    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": best_loss
        }, os.path.join(save_dir, "best_representation_model.pth"))

        tqdm.write(f"Best model saved at {epoch} epoch!")