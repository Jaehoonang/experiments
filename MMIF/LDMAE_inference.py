import torch
from models.LDMAE import LDMAE, DiffusionScheduler, compute_focus_score, focus_mask, apply_focus_mask
from data.dataset import ex_data1
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

@torch.no_grad()
def get_noise_prediction(model, x, t, cond):
    t_emb = model.time_mlp(t)

    z_diff = model.z_to_dit(x).unsqueeze(1)

    for dit_blk in model.dit_blocks:
        z_diff = dit_blk(z_diff, t_emb, cond=cond)

    z_diff = z_diff.squeeze(1)
    noise_pred = model.dit_to_noise(z_diff)

    return noise_pred

@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index, cond):
    betas_t = scheduler.extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = scheduler.extract(torch.sqrt(1.0 / scheduler.alphas), t, x.shape)

    predicted_noise = get_noise_prediction(model, x, t, cond)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = scheduler.extract(scheduler.betas, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, scheduler, shape, cond, device, save_interval=100):
    b = shape[0]

    img = torch.randn(shape, device=device)

    print("Sampling Start...")
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc='sampling loop', total=scheduler.timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        img = p_sample(model, scheduler, img, t, i, cond)
    return img

def decode_image(model, z, cond_feat, ids_restore, ids_mask):
    z_dec = model.from_latent(z)
    z_token = model.z_to_decoder(z_dec).unsqueeze(1)

    x_vis_emb = model.decoder_embed(cond_feat)

    B = x_vis_emb.shape[0]
    mask_tokens = model.mask_token.repeat(B, ids_mask.shape[1], 1)

    x_concat = torch.cat([x_vis_emb, z_token, mask_tokens], dim=1)

    z_tok = x_concat[:, x_vis_emb.shape[1]:x_vis_emb.shape[1] + 1]
    x_wo_z = torch.cat([x_concat[:, :x_vis_emb.shape[1]], x_concat[:, x_vis_emb.shape[1] + 1:]], dim=1)

    x_wo_z = torch.gather(
        x_wo_z,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, x_wo_z.shape[-1])
    )

    x_dec = torch.cat([z_tok, x_wo_z], dim=1)

    patch_tok = x_dec[:, 1:, :] + model.decoder_pos_embed
    x_dec = torch.cat([z_tok, patch_tok], dim=1)

    for blk in model.decoder_blocks:
        x_dec = blk(x_dec)

    x_dec = model.decoder_norm(x_dec)
    x_dec = x_dec[:, 1:, :]

    x_out = model.Decoder_pred(x_dec)
    x_out = model.unpatchify(x_out)

    return x_out

def stage1():
    vmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage1_vmae.pth"
    ldm_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_checkpoints\best_stage2_ldmae.pth"
    # 190001 210014 210016
    vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
    inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LDMAE(in_channels=1).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    checkpoint1 = torch.load(vmae_pt_path, map_location=device)
    checkpoint2 = torch.load(ldm_pt_path, map_location=device)
    model.load_state_dict(checkpoint1['model_state'], strict=False)
    model.eval()

    samples = []

    vis_image = ex_data1(root_dir=vis_img_path)
    inf_image = ex_data1(root_dir=inf_img_path)

    with torch.no_grad():
        x, mask, posterior = model(vis_image, inf_image)
        for _ in range(20):
            x, _, _ = model(vis_image, inf_image, sample_latent=True, latent_scale=30)
            samples.append(x.cpu())

    print(x.std(dim=1).mean())
    vis = vis_image.squeeze().cpu().numpy()
    inf = inf_image.squeeze().cpu().numpy()
    x_img = x.squeeze().cpu().numpy()
    x_img = np.clip(x_img, 0, 1)

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.title("Visible Input")
    plt.imshow(vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("Infrared Input")
    plt.imshow(inf, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.title("Summation feature")
    plt.imshow(inf+vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.title("Model Output (Reconstruction)")
    plt.imshow(x_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 8))
    for i, img in enumerate(samples):
        plt.subplot(4, 5, i + 1)
        plt.imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title(f"sample {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def stage2():
    vmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage1_vmae.pth"
    ldm_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_checkpoints\best_stage2_ldmae.pth"
    # 190001 210014 210016
    vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
    inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LDMAE(in_channels=1).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    checkpoint1 = torch.load(vmae_pt_path, map_location=device)
    checkpoint2 = torch.load(ldm_pt_path, map_location=device)
    model.load_state_dict(checkpoint2['model_state'])
    model.eval()

    samples = []

    vis_image = ex_data1(root_dir=vis_img_path)
    inf_image = ex_data1(root_dir=inf_img_path)

    with torch.no_grad():
        score = compute_focus_score(vis_image, inf_image, patch_size=model.Patch_Posi.patch_size)
        mask, ids_keep, ids_mask, ids_restore = focus_mask(score, mask_ratio=0.3)

        cond_patches = model.Patch_Posi(vis_image)
        cond_vis = apply_focus_mask(cond_patches, ids_keep)

        for blk in model.Encoder_blocks:
            cond_vis = blk(cond_vis)

        cond_feat = cond_vis

        latent_dim = 32
        latent_shape = (vis_image.shape[0], latent_dim)

        generated_z = p_sample_loop(model, scheduler, latent_shape, cond_feat, device)

        fused_image = decode_image(model, generated_z, cond_feat, ids_restore, ids_mask)

    vis_np = vis_image.squeeze().cpu().numpy()
    inf_np = inf_image.squeeze().cpu().numpy()
    fused_np = fused_image.squeeze().cpu().numpy()
    fused_np = np.clip(fused_np, 0, 1)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 4, 1)
    plt.title("Visible Input")
    plt.imshow(vis_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Infrared Input")
    plt.imshow(inf_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Summation feature")
    plt.imshow(inf_np + vis_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Model Output")
    plt.imshow(fused_np, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    stage2()

