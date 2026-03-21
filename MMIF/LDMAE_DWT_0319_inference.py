import torch
from models.LDMAE_DWT_0319 import LDMAE, DiffusionScheduler
from data.dataset import ex_data1, ex_data_dwt
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange

@torch.no_grad()
def get_noise_prediction(model, x_noisy_2d, t, cond_img_2d):
    noise_pred = model.unet_model(x=x_noisy_2d, timestep=t, cond=cond_img_2d)
    return noise_pred

@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index, cond):
    betas_t = scheduler.extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = scheduler.extract(torch.sqrt(1.0 / scheduler.alphas), t, x.shape)

    predicted_noise = get_noise_prediction(model, x, t, cond) # ids_keep 제거

    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = scheduler.extract(scheduler.betas, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, scheduler, shape, cond, device): # ids_keep 제거
    b = shape[0]
    img = torch.randn(shape, device=device) # shape은 이제 (B, 32, 28, 28)이 됩니다!

    print("Sampling Start...")
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc='sampling loop', total=scheduler.timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, scheduler, img, t, i, cond) # ids_keep 제거
    return img

def decode_image(model, generated_z, ids_restore, ids_mask):
    B = generated_z.shape[0]

    if generated_z.ndim == 3 and generated_z.shape[0] > B:
        generated_z = generated_z[-1]

    z_dec = model.latent_to_dec(generated_z)

    N_mask = ids_mask.shape[1]
    mask_tokens = model.mask_token.repeat(B, N_mask, 1)

    x_concat = torch.cat([z_dec, mask_tokens], dim=1)

    x_full = torch.gather(
        x_concat,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, x_concat.shape[-1])
    )

    x_full = x_full + model.decoder_pos_embed

    for blk in model.decoder_blocks:
        x_full = blk(x_full)

    x_full = model.decoder_norm(x_full)

    x_out = model.Decoder_pred(x_full)
    x_out = model.unpatchify(x_out)

    return x_out

# def decode_image(model, generated_z, cond_feat, ids_restore, ids_mask):
#     B = cond_feat.shape[0]
#
#     if generated_z.ndim == 3 and generated_z.shape[0] > B:
#         generated_z = generated_z[-1]
#
#     generated_z = generated_z.view(B, -1)
#
#     z_dec = model.latent_to_dec(generated_z)
#     z_token = model.z_to_decoder(z_dec).unsqueeze(1)
#
#     x_vis_emb = model.decoder_embed(cond_feat)
#
#     mask_tokens = model.mask_token.repeat(B, ids_mask.shape[1], 1)
#
#     x_concat = torch.cat([x_vis_emb, z_token, mask_tokens], dim=1)
#
#     z_tok = x_concat[:, x_vis_emb.shape[1]:x_vis_emb.shape[1] + 1]
#     x_wo_z = torch.cat([x_concat[:, :x_vis_emb.shape[1]], x_concat[:, x_vis_emb.shape[1] + 1:]], dim=1)
#
#     x_wo_z = torch.gather(
#         x_wo_z,
#         dim=1,
#         index=ids_restore.unsqueeze(-1).repeat(1, 1, x_wo_z.shape[-1])
#     )
#
#     x_dec = torch.cat([z_tok, x_wo_z], dim=1)
#
#     z_tok = x_dec[:, :1, :]
#     patch_tok = x_dec[:, 1:, :]
#     patch_tok = patch_tok + model.decoder_pos_embed
#
#     x_dec = torch.cat([z_tok, patch_tok], dim=1)
#
#     for blk in model.decoder_blocks:
#         x_dec = blk(x_dec)
#
#     x_dec = model.decoder_norm(x_dec)
#     x_dec = x_dec[:, 1:, :]
#     x_out = model.Decoder_pred(x_dec)
#     x_out = model.unpatchify(x_out)
#
#     return x_out

def stage1(freq='low'):
    # low_vmae_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_0312_checkpoints\best_stage1_low_vmae.pth"
    # high_vmae_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_0312_checkpoints\best_stage1_high_vmae.pth"

    low_vmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage1_low_vmae.pth"
    high_vmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage1_high_vmae.pth"

    # 190001 210014 210016
    # vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LLVIP\visible\test\210014.jpg"
    # inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LLVIP\infrared\test\210014.jpg"

    vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
    inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ch = 1 if freq == 'low' else 3

    model = LDMAE(in_channels=ch, img_size=112).to(device)

    if freq == 'low':
        checkpoint1 = torch.load(low_vmae_pt_path, map_location=device)

    else:
        checkpoint1 = torch.load(high_vmae_pt_path, map_location=device)

    filtered_state_dict = {k: v for k, v in checkpoint1['model_state'].items() if 'unet_model' not in k and 'cond_encoder_2d' not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    samples = []

    vis_image = ex_data1(root_dir=vis_img_path)
    inf_image = ex_data1(root_dir=inf_img_path)

    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)

    with torch.no_grad():
        vis_LL, vis_HF_list = dwt(vis_image)
        ir_LL, ir_HF_list = dwt(inf_image)

        if freq == 'low':
            img1_freq, img2_freq = vis_LL, ir_LL
        else:
            img1_freq = vis_HF_list[0].squeeze(2)
            img2_freq = ir_HF_list[0].squeeze(2)

        x, mask, posterior = model(img1_freq, img2_freq, stage=1)

        for _ in range(20):
            samp_x, _, _ = model(img1_freq, img2_freq, stage=1, sample_latent=True, latent_scale=30)
            samples.append(samp_x.cpu())

    if freq == 'low':
        vis = img1_freq.squeeze().detach().cpu().numpy()
        inf = img2_freq.squeeze().detach().cpu().numpy()
        x_img = x.squeeze().detach().cpu().numpy()
        x_img = np.clip(x_img, 0, 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title("Visible Input")
        plt.imshow(vis, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Infrared Input")
        plt.imshow(inf, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("Summation feature")
        plt.imshow(inf + vis, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Model Output (Recon)")
        plt.imshow(x_img, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 8))
        for i, img in enumerate(samples):
            plt.subplot(4, 5, i + 1)
            plt.imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
            plt.title(f"sample {i + 1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        low_feature = x_img
        return low_feature

    else:
        out_np = x.squeeze().detach().cpu().numpy()

        vh_np = np.abs(img1_freq.squeeze().detach().cpu().numpy())
        ih_np = np.abs(img2_freq.squeeze().detach().cpu().numpy())

        sum_np = vh_np + ih_np

        plt.figure(figsize=(20, 6))
        bands = ['LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']
        columns = ['Visible Input', 'Infrared Input', 'Summation Feature', 'Model Output']

        for i in range(3):
            plt.subplot(3, 4, i * 4 + 1)
            if i == 0: plt.title(columns[0])
            plt.imshow(vh_np[i], cmap='gray')
            plt.ylabel(bands[i])
            plt.xticks([])
            plt.yticks([])

            plt.subplot(3, 4, i * 4 + 2)
            if i == 0: plt.title(columns[1])
            plt.imshow(ih_np[i], cmap='gray')
            plt.axis('off')

            plt.subplot(3, 4, i * 4 + 3)
            if i == 0: plt.title(columns[2])
            plt.imshow(sum_np[i], cmap='gray')
            plt.axis('off')

            plt.subplot(3, 4, i * 4 + 4)
            if i == 0: plt.title(columns[3])
            plt.imshow(out_np[i], cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        high_feature = out_np
        return high_feature

def stage2(freq='low'):
    low_vmae_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_0319_checkpoints\best_stage2_low_ldmae.pth"
    high_vmae_pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\LDMAE_DWT_0319_checkpoints\best_stage2_high_ldmae.pth"

    # 190001 210014 210016
    vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
    inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ch = 1 if freq == 'low' else 3

    model = LDMAE(in_channels=ch, img_size=112).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    if freq == 'low':
        checkpoint2 = torch.load(low_vmae_pt_path, map_location=device)

    else:
        checkpoint2 = torch.load(high_vmae_pt_path, map_location=device)

    model.load_state_dict(checkpoint2['model_state'], strict=True)
    model.eval()

    vis_low, vis_high = ex_data_dwt(root_dir=vis_img_path)
    inf_low, inf_high = ex_data_dwt(root_dir=inf_img_path)

    if vis_low.max() > 1.0:
        vis_low = vis_low / 255.0
        vis_high = [h / 255.0 for h in vis_high]
        inf_low = inf_low / 255.0
        inf_high = [h / 255.0 for h in inf_high]

    vis_low = (vis_low - 0.5) / 0.5
    inf_low = (inf_low - 0.5) / 0.5

    if freq == 'low':
        vis_image = vis_low.to(device)
        inf_image = inf_low.to(device)
    else:
        vh = vis_high[0] if isinstance(vis_high, (list, tuple)) else vis_high
        ih = inf_high[0] if isinstance(inf_high, (list, tuple)) else inf_high
        vis_image = vh.squeeze(1).to(device)
        inf_image = ih.squeeze(1).to(device)

    with torch.no_grad():
        cond_feat = model.cond_encoder_2d(vis_image)

        B = vis_image.shape[0]
        grid_size = model.Patch_Posi.patch_size
        H = W = 112 // (grid_size[0] if isinstance(grid_size, tuple) else grid_size)  # 28
        D = model.enc_to_latent.out_features // 2  # 32

        latent_shape = (B, D, H, W)

        generated_z_2d = p_sample_loop(model, scheduler, latent_shape, cond_feat, device)

        scale_factor = 0.17545
        generated_z_2d = generated_z_2d / scale_factor

        generated_z_1d = rearrange(generated_z_2d, 'b c h w -> b (h w) c')

        z_dec = model.latent_to_dec(generated_z_1d)
        x_full = z_dec + model.decoder_pos_embed

        for blk in model.decoder_blocks:
            x_full = blk(x_full)

        x_full = model.decoder_norm(x_full)
        output = model.Decoder_pred(x_full)
        output = model.unpatchify(output)

    if freq == 'low':
        vis_np = vis_image.squeeze().cpu().numpy()
        inf_np = inf_image.squeeze().cpu().numpy()
        out_low = output.squeeze().cpu().numpy()
        out_low = (out_low * 0.5) + 0.5
        out_low = np.clip(out_low, 0, 1)

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
        plt.title("Diffusion Output")
        plt.imshow(out_low, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return out_low

    else:
        vh_np = vis_image.squeeze().cpu().numpy()
        ih_np = inf_image.squeeze().cpu().numpy()
        out_high = output.squeeze().cpu().numpy()

        sum_np = vh_np + ih_np

        plt.figure(figsize=(20, 6))
        bands = ['LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']
        columns = ['Visible Input', 'Infrared Input', 'Summation Feature', 'Diffusion Output']


        for i in range(3):
            plt.subplot(3, 4, i * 4 + 1)
            if i == 0: plt.title(columns[0])
            plt.imshow(vh_np[i], cmap='gray')
            plt.ylabel(bands[i])
            plt.xticks([])
            plt.yticks([])

            plt.subplot(3, 4, i * 4 + 2)
            if i == 0: plt.title(columns[1])
            plt.imshow(ih_np[i], cmap='gray')
            plt.axis('off')

            plt.subplot(3, 4, i * 4 + 3)
            if i == 0: plt.title(columns[2])
            plt.imshow(sum_np[i], cmap='gray')
            plt.axis('off')

            plt.subplot(3, 4, i * 4 + 4)
            if i == 0: plt.title(columns[3])
            plt.imshow(out_high[i], cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        return out_high

if __name__ == '__main__':

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
    low_feat = stage1(freq='low')
    # high_feat = stage1(freq='high')
    #
    # low_tensor = torch.tensor(low_feat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    # high_tensor = torch.tensor(high_feat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    #
    # ifm = DWTInverse(mode="periodization", wave='haar').to(device)
    #
    # inverse_out = ifm((low_tensor, [high_tensor]))
    #
    # final_img = inverse_out.squeeze().detach().cpu().numpy()
    # final_img = np.clip(final_img, 0, 1)
    #
    # plt.figure(figsize=(6, 6))
    # plt.title("Final Output Image(IDWT)")
    # plt.imshow(final_img, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # low_feat = stage2(freq='low')
    # high_feat = stage2(freq='high')
    #
    # low_tensor = torch.tensor(low_feat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    # high_tensor = torch.tensor(high_feat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    #
    # ifm = DWTInverse(mode="periodization", wave='haar').to(device)
    #
    # inverse_out = ifm((low_tensor, [high_tensor]))
    #
    # final_img = inverse_out.squeeze().detach().cpu().numpy()
    # final_img = np.clip(final_img, 0, 1)
    #
    # plt.figure(figsize=(6, 6))
    # plt.title("Final Output Image(IDWT)")
    # plt.imshow(final_img, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
