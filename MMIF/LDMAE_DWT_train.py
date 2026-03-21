import torch
from models.LDMAE_DWT import LDMAE, DiffusionScheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import VGG19_Weights, vgg19
from pathlib import Path
from torchvision import transforms
from data.VMAE_data import VMAE_Dataset
from utils.loss import GradientLoss
import os

def to_3ch(x):
    return x.repeat(1, 3, 1, 1)

def vgg_norm(x):
    return (x - mean) / std

def train_stage1_vmae(model, dataloader, epochs, device, save_dir, vgg, mean, std, grad_loss_fn, beta=1e-4, freq='low'):
    print("\nStarting Stage 1: Autoencoder (VMAE) Training...")

    stage1_params = [p for n, p in model.named_parameters() if 'dit' not in n and 'time_mlp' not in n]
    optimizer = torch.optim.AdamW(stage1_params, lr=1e-4)

    best_loss = float("inf")
    save_path = os.path.join(save_dir, f"best_stage1_{freq}_vmae.pth")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        epoch_bar = tqdm(dataloader, desc=f"Stage 1 [Epoch {epoch + 1}/{epochs}]", leave=False)

        for modal1_img, modal2_img in epoch_bar:
            modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)

            optimizer.zero_grad()

            output, mask, posterior, img1_freq, img2_freq = model(modal1_img, modal2_img, stage=1, freq=freq)

            target = torch.max(img1_freq, img2_freq)

            B, C, H, W = output.shape
            p = model.Patch_Posi.patch_size
            if isinstance(p, tuple): p = p[0]

            h, w = H // p, W // p

            if mask.ndim == 3 and mask.shape[1] == C:
                img_mask = mask.reshape(B, C, h, w)
            else:
                img_mask = mask.reshape(B, 1, h, w).expand(B, C, h, w)

            img_mask = img_mask.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)
            visible_mask = 1 - img_mask

            recon = F.l1_loss(output, target, reduction='none')

            recon_loss1 = (recon * img_mask).sum() / (img_mask.sum() + 1e-6)
            recon_loss2 = (recon * visible_mask).sum() / (visible_mask.sum() + 1e-6)

            with torch.no_grad():
                target_vgg = target if C == 3 else to_3ch(target)

            output_vgg = output if C == 3 else to_3ch(output)
            perceptual_loss = F.l1_loss(vgg(vgg_norm(output_vgg)), vgg(vgg_norm(target_vgg)))

            grad_loss = grad_loss_fn(output, img1_freq, img2_freq)

            mu = posterior.mean
            logvar = posterior.logvar
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = (2 * recon_loss1) + recon_loss2 + perceptual_loss + beta * kl_loss + grad_loss

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

def train_stage2_diffusion(model, dataloader, epochs, device, save_dir, scheduler, stage1_weights_path, freq='low'):
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
    save_path = os.path.join(save_dir, f"best_stage2_{freq}_ldmae.pth")

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

            noise_target, noise_pred = model(modal1_img, modal2_img, stage=2, freq=freq, timestep=t, scheduler=scheduler)
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

    low_model = LDMAE(in_channels=1, img_size=112).to(device)
    high_model = LDMAE(in_channels=3, img_size=112).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:14].to(device)
    vgg.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    grad_loss_fn = GradientLoss().to(device)

    save_dir = "LDMAE_DWT_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    low_pre_model_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_checkpoints\best_stage1_low_vmae.pth"
    high_pre_model_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_checkpoints\best_stage1_high_vmae.pth"
    stage1_epochs = 1000
    stage2_epochs = 500

    stage1_weights = train_stage1_vmae(
        model=low_model,
        dataloader=dataloader,
        epochs=stage1_epochs,
        device=device,
        save_dir=save_dir,
        vgg=vgg,
        mean=mean,
        std=std,
        grad_loss_fn=grad_loss_fn,
        freq='low'
    )

    train_stage2_diffusion(
        model=low_model,
        dataloader=dataloader,
        epochs=stage2_epochs,
        device=device,
        save_dir=save_dir,
        scheduler=scheduler,
        stage1_weights_path=stage1_weights,
        freq='low'
    )

    print("All training stages completed!")