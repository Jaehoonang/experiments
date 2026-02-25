from models.LDM import SimpleUNet, DiffusionScheduler, SimpleUNetWithAttention
from data.dataset import ex_data1, ex_data2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#
#
# diff_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDM_checkpoints\best_ldm_model.pth"
# # vmae_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\VMAE_checkpoints\best_representation_model.pth"
#
# vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
# inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# diff_model = DiffUNet(dim=64, channels=1, dim_mults=(1, 2, 4, 8)).to(device)
# # vmae_model = VMAE(in_channels=1, patch_size=16).to(device)
#
# diff_model.eval()
# diff_checkpoint = torch.load(diff_pt_path, map_location=device)
# diff_model.load_state_dict(diff_checkpoint['model_state'])
#
# # vmae_model.eval()
# # vmae_checkpoint = torch.load(vmae_pt_path, map_location=device)
# # vmae_model.load_state_dict(vmae_checkpoint['model_state'])
#
# vis_image = ex_data2(root_dir=vis_img_path)
# inf_image = ex_data2(root_dir=inf_img_path)
#
# reverse_transform = Compose([
#      Lambda(lambda t: (t + 1) / 2),
#      Lambda(lambda t: t.permute(1, 2, 0)),# CHW to HWC
#      Lambda(lambda t: t * 255.),
#      Lambda(lambda t: t.numpy().astype(np.uint8)),
#      ToPILImage(),
# ])
#
# timesteps = 500
# betas = cosine_beta_schedule(timesteps).to(device)
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
# save_interval = 100
#
# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = a.gather(-1, t)
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
#
# @torch.no_grad()
# def p_sample(model, x, t, t_index, cond):
#     betas_t = extract(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
#     sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
#
#     predicted_noise = model(x, t, cond)
#     print("pred noise:", predicted_noise.mean().item(), predicted_noise.std().item())
#     model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
#
#     if t_index == 0:
#         return model_mean
#
#     else:
#         posterior_variance_t = extract(posterior_variance, t, x.shape)
#         noise = torch.randn_like(x)
#         a = model_mean + torch.sqrt(posterior_variance_t) * noise
#         print("x_t:", a.mean().item(), a.std().item())
#         a = torch.clamp(a, -1, 1)
#         return a
#
# @torch.no_grad()
# def p_sample_loop(model, shape, cond):
#     device = next(model.parameters()).device
#
#     b = shape[0]
#     hint_lambda = 0
#     img = (1 - hint_lambda) * torch.randn(shape, device=device) + hint_lambda * cond
#     imgs = []
#     steps = []
#
#     for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
#         img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, cond)
#
#         if i % save_interval == 0:
#             imgs.append(img.cpu())
#             steps.append(i)
#
#     imgs.append(img.cpu())
#     steps.append(0)
#
#     return imgs, steps
#
# @torch.no_grad()
# def sample(model, cond, image_size=224, batch_size=1, channels=1):
#     return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), cond=cond)
#
# sampled_img, steps = sample(diff_model, cond=vis_image)
# # output_img = reverse_transform(sampled_img[-1][0].cpu())
#
# samples = sampled_img
# final = sampled_img[-1][0]
# print(final.min(), final.max(), final.mean())
#
# plt.figure(figsize=(20,4))
# for i, img in enumerate(samples):
#     plt.subplot(1, len(samples), i+1)
#
#     vis = reverse_transform(img[0])
#     plt.title(f"t={steps[i]}")
#
#     plt.imshow(vis, cmap='gray')
#     plt.axis("off")
# plt.tight_layout()
# plt.show()

##################################################################################################################################################

@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index, cond):
    # 1. 필요한 계수들 추출
    betas_t = scheduler.extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = scheduler.extract(
        torch.sqrt(1.0 / scheduler.alphas), t, x.shape
    )

    predicted_noise = model(x, t, cond)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:

        posterior_variance_t = scheduler.extract(scheduler.betas.to(device), t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, scheduler, shape, cond, save_interval=100):
    """
    전체 샘플링 과정 실행 및 중간 결과 저장
    """
    device = next(model.parameters()).device
    b = shape[0]

    # 1. 가우시안 노이즈에서 시작 (x_T ~ N(0, 1))
    img = torch.randn(shape, device=device)
    imgs = [img]  # 초기 노이즈 저장
    steps = [scheduler.timesteps]


    # 2. T-1부터 0까지 거꾸로 반복 (Reverse Process)
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc='sampling loop time step', total=scheduler.timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        # 이미지 갱신 (x_t -> x_{t-1})
        img = p_sample(model, scheduler, img, t, i, cond)
        img = torch.clamp(img, -1.0, 1.0)

        if i % save_interval == 0 or i == 0:
            imgs.append(img.detach().cpu())
            steps.append(i)

    return imgs, steps  # 모든 타임스텝의 이미지 리스트 반환

@torch.no_grad()
def ddim_sample_loop(model, scheduler, shape, cond, eta=0.0):
    """
    DDIM Sampling (Deterministic)
    eta=0.0 이면 완전한 결정론적 샘플링 (노이즈가 가장 적음)
    """
    device = next(model.parameters()).device
    b = shape[0]

    # 1. 랜덤 노이즈에서 시작
    img = torch.randn(shape, device=device)

    # 결과 저장용
    imgs = []

    print("Starting DDIM Sampling...")

    # 타임스텝 루프 (reversed)
    for i in tqdm(reversed(range(0, scheduler.timesteps)), total=scheduler.timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        # 1. 현재 시점의 변수들 추출
        # alpha_cumprod (bar_alpha)
        alpha_bar_t = scheduler.extract(scheduler.alphas_cumprod.to(device), t, img.shape)

        # 이전 시점(t-1)의 alpha_cumprod 추출 (t=0일 때는 1.0으로 설정)
        if i == 0:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            prev_t = torch.full((b,), i - 1, device=device, dtype=torch.long)
            alpha_bar_prev = scheduler.extract(scheduler.alphas_cumprod.to(device), prev_t, img.shape)

        # 2. 모델이 노이즈 예측 (epsilon_theta)
        pred_noise = model(img, t, cond)

        # 3. 예측된 x_0 추정 (Clean Image Prediction)
        # 수식: (x_t - sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
        pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        # [중요] x_0를 -1 ~ 1로 Clipping (이것이 노이즈 제거에 핵심)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 4. x_{t-1} 계산 (DDIM 공식)
        # Direction pointing to x_t
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))

        # x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + direction + random_noise
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * pred_noise

        noise = torch.randn_like(img) if eta > 0 else 0.

        img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

    return [img.cpu()], [0]  # 최종 결과만 리스트로 반환

@torch.no_grad()
def visualize_reverse_process(model, scheduler, shape=(1, 1, 28, 28)):
    model.eval()
    # 1. 샘플 생성 (imgs[0]은 노이즈, imgs[-1]은 깨끗한 이미지)
    samples = p_sample_loop(model, scheduler, shape)

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

# pred_noise = model(img, t, visible)

device = "cuda" if torch.cuda.is_available() else "cpu"

vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"
vis_image = ex_data2(root_dir=vis_img_path)
inf_image = ex_data2(root_dir=inf_img_path)

model = SimpleUNet().to(device)
# model = SimpleUNetWithAttention().to(device)
ckpt = torch.load("LDM_checkpoints/best_representation_model.pth")
model.load_state_dict(ckpt["model_state"])
model.eval()

cond = inf_image
cond = torch.clamp(cond, -1.0, 1.0)

print("cond min/max:", cond.min().item(), cond.max().item())
print("cond mean/std:", cond.mean().item(), cond.std().item())
shape = cond.shape

scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')
samples, steps = p_sample_loop(model, scheduler, shape, cond, save_interval=100)
# samples, steps = ddim_sample_loop(model, scheduler, shape, cond, eta=0.0)

plt.figure(figsize=(15,5))
for i in range(len(samples)):
    plt.subplot(1, len(samples), i+1)
    img = samples[i][0].squeeze().detach().cpu().numpy()
    img = (img + 1) / 2
    print(img.min(), img.max())
    plt.imshow(img, cmap='gray')
    plt.title(f"t={steps[i]}")
    plt.axis('off')

plt.show()