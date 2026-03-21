import torch
from models.LDMAE_GServer import LDMAE, DiffusionScheduler
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import VGG19_Weights, vgg19
from pathlib import Path
from torchvision.transforms import Compose, Lambda, ToPILImage
from torchvision import transforms
from data.VMAE_data import VMAE_Dataset
from utils.loss import GradientLoss
import os

def to_3ch(x):
    return x.repeat(1, 3, 1, 1)

def vgg_norm(x):
    return (x - mean) / std

# beta = 1e-4
# epochs = 2000
# save_dir = "LDMAE_checkpoints"
# os.makedirs(save_dir, exist_ok=True)
# grad_loss_fn = GradientLoss()
# best_loss = float("inf")
#
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0.0
#
#     epoch_bar = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]", leave=False)
#
#     for modal1_img, modal2_img in epoch_bar:
#         modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)
#
#         optimizer.zero_grad()
#
#         output, mask, posterior = model(modal1_img, modal2_img)
#         target = torch.max(modal1_img, modal2_img)
#
#         patch_mask = mask  #
#         B, _, H, W = output.shape
#         p = model.Patch_Posi.patch_size
#         h = H // p
#         w = W // p
#         img_mask = patch_mask.reshape(B, h, w)
#         img_mask = img_mask.repeat_interleave(p, 1).repeat_interleave(p, 2)
#         img_mask = img_mask.unsqueeze(1)  # [B,1,H,W]
#
#         recon_loss = torch.abs(output - target)
#         recon_loss = (recon_loss * img_mask).sum() / (img_mask.sum() + 1e-6)
#
#
#         with torch.no_grad():
#             target_feat = vgg(vgg_norm(to_3ch(torch.max(modal1_img, modal2_img))))
#
#         output_feat = vgg(vgg_norm(to_3ch(output)))
#         perceptual_loss = F.l1_loss(output_feat, target_feat)
#
#         grad_loss = grad_loss_fn(output, modal1_img, modal2_img)
#
#         # KL loss
#         mu = posterior.mean
#         logvar = posterior.logvar
#
#         kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#
#         loss = recon_loss + perceptual_loss + beta*kl_loss + grad_loss
#
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         epoch_bar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", percepual=f"{perceptual_loss.item():.4f}", KL=f"{kl_loss.item():.4f}", grad=f"{grad_loss.item():.4f}")
#     avg_loss = epoch_loss / len(dataloader)
#
#     tqdm.write(f"Epoch [{epoch + 1}/{epochs}] | Reconstruction Loss: {recon_loss:.4f} | Perceptual Loss: {perceptual_loss:.4f} | KL Loss: {kl_loss:.4f} | grad Loss: {grad_loss:.4f} | Avg Loss: {avg_loss:.4f}")
#
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save({
#             "epoch": epoch + 1,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "loss": best_loss
#         }, os.path.join(save_dir, f"best_representation_model.pth"))
#
#         tqdm.write(f"Best model saved at {epoch + 1} epoch!")


def train_stage1_vmae(model, dataloader, epochs, device, save_dir, vgg, mean, std, grad_loss_fn, beta=1e-4):
    print("\nStarting Stage 1: Autoencoder (VMAE) Training...")

    stage1_params = [p for n, p in model.named_parameters() if 'dit' not in n and 'time_mlp' not in n]
    optimizer = torch.optim.AdamW(stage1_params, lr=1e-4)

    best_loss = float("inf")
    save_path = os.path.join(save_dir, "best_stage1_vmae.pth")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        epoch_bar = tqdm(dataloader, desc=f"Stage 1 [Epoch {epoch + 1}/{epochs}]", leave=False)

        for modal1_img, modal2_img in epoch_bar:
            modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)
            target = torch.max(modal1_img, modal2_img)

            optimizer.zero_grad()

            output, mask, posterior = model(modal1_img, modal2_img, stage=1)

            B, _, H, W = output.shape
            p = model.Patch_Posi.patch_size
            if isinstance(p, tuple): p = p[0]

            h, w = H // p, W // p
            img_mask = mask.reshape(B, h, w)
            img_mask = img_mask.repeat_interleave(p, 1).repeat_interleave(p, 2).unsqueeze(1)
            visible_mask = 1 - img_mask

            recon = F.l1_loss(output, target, reduction='none')
            recon_loss1 = (recon * img_mask).sum() / (img_mask.sum() + 1e-6)

            recon_loss2 = (recon * visible_mask).sum() / (visible_mask.sum() + 1e-6)

            with torch.no_grad():
                target_feat = vgg(vgg_norm(to_3ch(torch.max(modal1_img, modal2_img))))

            output_feat = vgg(vgg_norm(to_3ch(output)))
            perceptual_loss = F.l1_loss(output_feat, target_feat)

            grad_loss = grad_loss_fn(output, modal1_img, modal2_img)

            mu = posterior.mean
            logvar = posterior.logvar
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss1 + recon_loss2 + perceptual_loss + beta * kl_loss + grad_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bar.set_postfix(Loss=f"{loss.item():.4f}", Recon1=f"{recon_loss1.item():.4f}", Recon2=f"{recon_loss2.item():.4f}",
                                  Percept=f"{perceptual_loss.item():.4f}", KL=f"{kl_loss.item():.4f}",
                                  Grad=f"{grad_loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        tqdm.write(
            f"Stage 1 Epoch [{epoch + 1}/{epochs}] | Recon1: {recon_loss1:.4f} | Recon2: {recon_loss2:.4f} |Perceptual: {perceptual_loss:.4f} | KL: {kl_loss:.4f} | Grad: {grad_loss:.4f} | Avg: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss
            }, save_path)
            tqdm.write(f" Best Stage 1 model saved at {epoch + 1} epoch!")

    return save_path


def train_stage2_diffusion(model, dataloader, epochs, device, save_dir, scheduler, stage1_weights_path):
    print("\n Starting Stage 2: Diffusion Transformer (DiT) Training...")

    if os.path.exists(stage1_weights_path):
        checkpoint = torch.load(stage1_weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("Pre-trained Stage 1 weights loaded successfully!")
    else:
        print("Warning: Stage 1 weights not found. Training from scratch.")

    for name, param in model.named_parameters():
        if 'dit' not in name and 'time_mlp' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    stage2_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(stage2_params, lr=1e-4)

    best_loss = float("inf")
    save_path = os.path.join(save_dir, "best_stage2_ldmae.pth")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        epoch_bar = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]", leave=False)

        for modal1_img, modal2_img in epoch_bar:
            modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)

            optimizer.zero_grad()

            x_0 = (modal1_img + modal2_img) / 2
            batch_size = x_0.shape[0]

            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()

            noise_target, noise_pred = model(modal1_img, modal2_img, stage=2, timestep=t, scheduler=scheduler)
            loss = F.mse_loss(noise_pred, noise_target)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)

        tqdm.write(
            f"Epoch [{epoch + 1}/{epochs}] | MSE Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss
            }, save_path)

            tqdm.write(f" Best model saved at {epoch + 1} epoch!")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"현재 사용 중인 device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\infrared")
    dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\visible")
    dataset = VMAE_Dataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LDMAE(in_channels=1).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:14].to(device)
    vgg.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    grad_loss_fn = GradientLoss().to(device)

    save_dir = "LDMAE_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    pre_model_path = r"C:\Users\12wkd\Desktop\best_stage1_vmae.pth"
    stage1_epochs = 3
    stage2_epochs = 300

    # stage1_weights = train_stage1_vmae(
    #     model=model,
    #     dataloader=dataloader,
    #     epochs=stage1_epochs,
    #     device=device,
    #     save_dir=save_dir,
    #     vgg=vgg,
    #     mean=mean,
    #     std=std,
    #     grad_loss_fn=grad_loss_fn
    # )

    train_stage2_diffusion(
        model=model,
        dataloader=dataloader,
        epochs=stage2_epochs,
        device=device,
        save_dir=save_dir,
        scheduler=scheduler,
        stage1_weights_path=pre_model_path
    )

    print("All training stages completed!")