from models.LDM import SimpleUNet, DiffusionScheduler, SimpleUNetWithAttention
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

from pathlib import Path
from torchvision.transforms import Compose, Lambda, ToPILImage
from data.VMAE_data import VMAE_Dataset
import torch
import os
import matplotlib.pyplot as plt

#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"현재 사용 중인 device {device}")
#
# timesteps = 500
# betas = cosine_beta_schedule(timesteps).to(device)
#
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, dim=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
# sqrt_recip_alphas_cumprod_minus_one = torch.sqrt(1. / alphas_cumprod - 1)
#
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
#
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#
# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = a.gather(-1, t)
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
#
# def q_sample(x_start, t, noise=None):
#     if noise is None:
#         noise = torch.randn_like(x_start)
#
#     sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
#
#     return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
#
# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     Lambda(lambda t: (t * 2) - 1),
# ])
#
# def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
#     if noise is None:
#         noise = torch.randn_like(x_start)
#
#     x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
#     predicted_noise = denoise_model(x_noisy, t)
#
#     if loss_type == 'l1':
#         loss = F.l1_loss(noise, predicted_noise)
#     elif loss_type == 'l2':
#         loss = F.mse_loss(noise, predicted_noise)
#     elif loss_type == "huber":
#         loss = F.smooth_l1_loss(noise, predicted_noise)
#     else:
#         raise NotImplementedError()
#
#     return loss
#
# def num_to_groups(num, divisor):
#     groups = num // divisor
#     remainder = num % divisor
#     arr = [divisor] * groups
#     if remainder > 0:
#         arr.append(remainder)
#     return arr
#
#
# dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\infrared")
# dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\visible")
#
# dataset= VMAE_Dataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# model = DiffUNet(dim=64, channels=1, dim_mults=(1, 2, 4, 8)).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#
# epochs = 300
# save_dir = "LDM_checkpoints"
# os.makedirs(save_dir, exist_ok=True)
# best_loss = float("inf")
#
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0.0
#
#     epoch_bar = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]", leave=False)
#
#     for modal1_img, modal2_img in epoch_bar:
#         modal1_img = modal1_img.to(device)
#         modal2_img = modal2_img.to(device)
#
#         cond = modal1_img
#
#         optimizer.zero_grad()
#
#         x_start = torch.abs(modal1_img) + torch.abs(modal2_img)
#
#         B = x_start.shape[0]
#         t = torch.randint(0, timesteps, (B,), device=device).long()
#
#         x_noisy, noise = q_sample(x_start, t)
#
#         predicted_noise = model(x_noisy, t, cond=cond)
#
#         loss = F.mse_loss(predicted_noise, noise, reduction='sum')
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#         epoch_bar.set_postfix(loss=f"{loss.item():.4f}")
#
#     avg_loss = epoch_loss / len(dataloader)
#
#     tqdm.write(f"Epoch [{epoch + 1}/{epochs}] | Avg Loss: {avg_loss:.4f}")
#
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save({
#             "epoch": epoch + 1,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "loss": best_loss
#         }, os.path.join(save_dir, "best_ldm_model.pth"))
#
#         tqdm.write(f"Best model saved at {epoch + 1} epoch!")
########################################################################################################################################################
@torch.no_grad()
def sample(model, scheduler, shape):
    device = next(model.parameters()).device
    b = shape[0]

    img = torch.randn(shape, device=device)

    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        betas_t = scheduler.extract(scheduler.betas, t, img.shape)
        sqrt_one_minus = scheduler.extract(
            scheduler.sqrt_one_minus_alphas_cumprod, t, img.shape
        )
        sqrt_recip = scheduler.extract(
            torch.sqrt(1.0 / scheduler.alphas), t, img.shape
        )

        pred_noise = model(img, t)

        model_mean = sqrt_recip * (img - betas_t * pred_noise / sqrt_one_minus)

        if i > 0:
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(betas_t) * noise
        else:
            img = model_mean

    return img

@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index, cond):
    """
    모델의 예측값을 이용해 x_t에서 x_{t-1}을 계산 (Sampling 한 단계)
    """
    # 1. 필요한 계수들 추출
    betas_t = scheduler.extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = scheduler.extract(
        torch.sqrt(1.0 / scheduler.alphas), t, x.shape
    )

    # 2. 모델이 노이즈 예측 (epsilon_theta)
    predicted_noise = model(x, t, cond)

    # 3. x_{t-1}의 평균 계산 (DDPM 공식 11번)
    # model_mean = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon_theta)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        # 4. t > 0일 때 노이즈(sigma_t * z) 추가 (Langevin dynamics)
        # DDPM 논문에 따라 sigma_t^2 = beta_t 로 설정하는 것이 일반적입니다.
        posterior_variance_t = scheduler.extract(scheduler.betas, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, scheduler, shape, cond):
    """
    전체 샘플링 과정 실행 및 중간 결과 저장
    """
    device = next(model.parameters()).device
    b = shape[0]

    # 1. 가우시안 노이즈에서 시작 (x_T ~ N(0, 1))
    img = torch.randn(shape, device=device)
    imgs = [img]  # 초기 노이즈 저장

    # 2. T-1부터 0까지 거꾸로 반복 (Reverse Process)
    for i in reversed(range(0, scheduler.timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        # 이미지 갱신 (x_t -> x_{t-1})
        img = p_sample(model, scheduler, img, t, i, cond)

        # 시각화를 위해 중간 결과들 저장
        imgs.append(img)

    return imgs  # 모든 타임스텝의 이미지 리스트 반환

@torch.no_grad()
def visualize_reverse_process(model, scheduler, shape=(1, 1, 28, 28)):
    model.eval()
    # 1. 샘플 생성 (imgs[0]은 노이즈, imgs[-1]은 깨끗한 이미지)
    vis, inf = next(iter(dataloader))
    vis = vis.to(device)

    cond = vis
    shape = vis.shape

    samples = p_sample_loop(model, scheduler, shape, cond)

    total_steps = len(samples)  # 1001개 (T=1000 기준)

    # 2. 10개의 지점 선택 (0, 111, 222 ... 1000)
    # 인덱스 0이 노이즈(Step 1000)이고, 마지막 인덱스가 결과물(Step 0)임을 명심하세요.
    indices = np.linspace(0, total_steps - 1, 10, dtype=int)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]

        # samples[idx] 추출
        img = samples[idx][0].squeeze().cpu().numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        ax.imshow(img, cmap='gray')

        # 실제 타임스텝 t 계산:
        # idx가 0일 때 실제 t=1000 (노이즈)
        # idx가 1000일 때 실제 t=0 (결과물)
        actual_t = scheduler.timesteps - idx
        ax.set_title(f"Step {actual_t}")
        ax.axis('off')

    plt.suptitle("Reverse Diffusion Process (Noise to Image)", fontsize=16)
    plt.show()

def train_one_epoch(model, scheduler, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (vis, inf) in enumerate(dataloader):
        optimizer.zero_grad()

        vis = vis.to(device)
        inf = inf.to(device)

        x_0 = vis + inf
        batch_size = x_0.shape[0]

        # 무작위 타임스텝 t 샘플링
        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()

        # 노이즈 생성 및 정방향 확산
        noise = torch.randn_like(x_0)
        x_noisy = scheduler.q_sample(x_start=x_0, t=t, noise=noise)

        # 모델의 노이즈 예측 및 Loss 계산
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)

        total_loss += loss.item()

        # 역전파
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch [{epoch}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(dataloader)
    print(f">>> Epoch {epoch} Average Train Loss: {avg_train_loss:.4f} <<<\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 사용 중인 device {device}")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    Lambda(lambda t: (t * 2) - 1),
])

model = SimpleUNet().to(device)
# model = SimpleUNetWithAttention().to(device)
scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dal1 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\infrared")
dal2 = Path(r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\train\visible")

dataset= VMAE_Dataset(modal1_dir=dal1, modal2_dir=dal2, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

epochs = 200
save_dir = "LDM_checkpoints"
os.makedirs(save_dir, exist_ok=True)
best_loss = float("inf")

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

        # 노이즈 생성 및 정방향 확산
        noise = torch.randn_like(x_0)
        x_noisy = scheduler.q_sample(x_start=x_0, t=t, noise=noise)

        # 모델의 노이즈 예측 및 Loss 계산
        predicted_noise = model(x_noisy, t, cond=modal2_img)
        loss = F.mse_loss(noise, predicted_noise)

        epoch_loss += loss.item()

        # 역전파
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
        }, os.path.join(save_dir, f"best_representation_model.pth"))

        tqdm.write(f"Best model saved at {epoch + 1} epoch!")





