import torch

from models.LDMAE_DWT_0319 import LDMAE, DiffusionScheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import VGG19_Weights, vgg19
from pathlib import Path
from torchvision import transforms
from data.VMAE_data import VMAE_Dataset
from utils.loss import GradientLoss
import os
from pytorch_wavelets import DWTForward
from einops import rearrange

def to_3ch(x):
    return x.repeat(1, 3, 1, 1)

def vgg_norm(x):
    return (x - mean) / std

def train_stage1_vmae(model, dataloader, epochs, device, save_dir, vgg, mean, std, grad_loss_fn, dwt, beta=1e-4, freq='low', resume_path=None):
    print(f"\nStarting Stage 1: Autoencoder (VMAE) Training for {freq}...")

    stage1_params = [p for n, p in model.named_parameters() if 'unet_model' not in n and 'cond_encoder_2d' not in n]
    optimizer = torch.optim.AdamW(stage1_params, lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0
    save_path = os.path.join(save_dir, f"best_stage1_{freq}_vmae.pth")

    if resume_path is not None:
        if os.path.exists(resume_path):
            print(f"이전 체크포인트 불러오는 중: {resume_path} ...")
            checkpoint = torch.load(resume_path, map_location=device)

            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint['model_state'])

            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float("inf"))

            print(f"성공적으로 불러왔습니다! (시작 에폭: {start_epoch + 1}, 이전 베스트 Loss: {best_loss:.4f})")
        else:
            print(f"경고: 체크포인트 파일을 찾을 수 없습니다 ({resume_path}). 처음부터 학습을 시작합니다.")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bar = tqdm(dataloader, desc=f"Stage 1 [Epoch {epoch + 1}/{epochs}]", leave=False)

        for modal1_img, modal2_img in epoch_bar:
            modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)

            with torch.no_grad():
                vis_LL, vis_HF_list = dwt(modal1_img)
                ir_LL, ir_HF_list = dwt(modal2_img)

                if freq == 'low':
                    img1_freq, img2_freq = vis_LL, ir_LL
                else:
                    img1_freq = vis_HF_list[0].squeeze(2)
                    img2_freq = ir_HF_list[0].squeeze(2)

            optimizer.zero_grad()

            output, mask, posterior = model(img1_freq, img2_freq, stage=1)

            target = torch.max(img1_freq, img2_freq)

            if target.ndim == 5:
                target = rearrange(target, 'b c d h w -> b (c d) h w')

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

            img1_freq_loss = rearrange(img1_freq, 'b c d h w -> b (c d) h w') if img1_freq.ndim == 5 else img1_freq
            img2_freq_loss = rearrange(img2_freq, 'b c d h w -> b (c d) h w') if img2_freq.ndim == 5 else img2_freq

            grad_loss = grad_loss_fn(output, img1_freq_loss, img2_freq_loss)

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

def train_stage2_diffusion(model, dataloader, epochs, device, save_dir, scheduler, stage1_weights_path, dwt, freq='low', resume_path=None):
    print("\n Starting Stage 2: Diffusion Transformer (DiT) Training...")

    if os.path.exists(stage1_weights_path):
        checkpoint = torch.load(stage1_weights_path, map_location=device)
        filtered_state_dict = {k: v for k, v in checkpoint['model_state'].items() if 'unet_model' not in k and 'cond_encoder_2d' not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Pre-trained Stage 1 weights loaded successfully!")
    else:
        print("Warning: Stage 1 weights not found. Training from scratch.")

    for name, param in model.named_parameters():
        if 'unet_model' in name or 'cond_encoder_2d' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    stage2_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(stage2_params, lr=1e-4)

    best_loss = float("inf")
    save_path = os.path.join(save_dir, f"best_stage2_{freq}_ldmae.pth")

    if resume_path is not None:
        if os.path.exists(resume_path):
            print(f"이전 체크포인트 불러오는 중: {resume_path} ...")
            checkpoint = torch.load(resume_path, map_location=device)

            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint['model_state'])

            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float("inf"))

            print(f"성공적으로 불러왔습니다! (시작 에폭: {start_epoch + 1}, 이전 베스트 Loss: {best_loss:.4f})")
        else:
            print(f"경고: 체크포인트 파일을 찾을 수 없습니다 ({resume_path}). 처음부터 학습을 시작합니다.")

    for epoch in range(epochs):
        model.train()

        model.Patch_Posi.eval()
        model.Encoder_blocks.eval()
        model.enc_to_latent.eval()
        model.pos_to_latent.train()
        model.cond_encoder_2d.train()

        epoch_loss = 0.0

        epoch_bar = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]", leave=False)

        for modal1_img, modal2_img in epoch_bar:
            modal1_img, modal2_img = modal1_img.to(device), modal2_img.to(device)


            # print(f"\n[Condition 범위 확인] Min: {modal1_img.min().item():.4f}, Max: {modal1_img.max().item():.4f}")

            with torch.no_grad():
                vis_LL, vis_HF_list = dwt(modal1_img)
                ir_LL, ir_HF_list = dwt(modal2_img)

                if freq == 'low':
                    img1_freq, img2_freq = vis_LL, ir_LL
                else:
                    img1_freq = vis_HF_list[0].squeeze(2)
                    img2_freq = ir_HF_list[0].squeeze(2)

            optimizer.zero_grad()
            # with torch.no_grad():
            #     # 모델에 stage=1을 주면 VMAE가 알아서 계산 후 posterior를 반환합니다.
            #     _, _, posterior = model(img1_freq, img2_freq, stage=1)
            #     z_sample = posterior.sample()
            #
            #     print(f"\n🚨 [긴급 진단] VMAE z 평균: {z_sample.mean().item():.4f}, 표준편차: {z_sample.std().item():.4f}")
            #     exit()

            # =================================================================

            x_0 = (modal1_img + modal2_img) / 2
            batch_size = x_0.shape[0]

            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()

            noise_target, noise_pred = model(img1_freq, img2_freq, stage=2, timestep=t, scheduler=scheduler, sample_latent=False)
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

    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\infrared")
    dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\visible")
    dataset = VMAE_Dataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    low_model = LDMAE(in_channels=1, img_size=112).to(device)
    high_model = LDMAE(in_channels=3, img_size=112).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:14].to(device)
    vgg.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    grad_loss_fn = GradientLoss().to(device)

    save_dir = "LDMAE_DWT_0319_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    # low_pre_model_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_0319_checkpoints\best_stage1_low_vmae.pth"
    # high_pre_model_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_0319_checkpoints\best_stage1_high_vmae.pth"

    low_pre_model_path = r"C:\Users\12wkd\Desktop\best_stage1_low_vmae.pth"
    high_pre_model_path = r"C:\Users\12wkd\Desktop\best_stage1_high_vmae.pth"

    stage1_epochs = 1000
    stage2_epochs = 1000

    # stage1_weights = train_stage1_vmae(
    #     model=high_model,
    #     dataloader=dataloader,
    #     epochs=stage1_epochs,
    #     device=device,
    #     save_dir=save_dir,
    #     vgg=vgg,
    #     mean=mean,
    #     std=std,
    #     grad_loss_fn=grad_loss_fn,
    #     dwt=dwt,
    #     freq='high',
    #     resume_path=None
    # )

    train_stage2_diffusion(
        model=low_model,
        dataloader=dataloader,
        epochs=stage2_epochs,
        device=device,
        save_dir=save_dir,
        scheduler=scheduler,
        stage1_weights_path=low_pre_model_path,
        dwt=dwt,
        freq='low',
        resume_path=None
    )
    # scale factor 변경
    print("All training stages completed!")