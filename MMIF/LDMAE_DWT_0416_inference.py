import torch
from models.LDMAE_DWT_0416 import LDMAE, DiffusionScheduler

from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange
from MMIF.utils.misc import DiagonalGaussianDistribution
import torch.nn.functional as F
from data.dataset import ex_data1, ex_data_dwt
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

def patchify_focus(img, patch_size):
    if img.ndim == 5:
        img = rearrange(img, 'b c d h w -> b (c d) h w')
    B, C, H, W = img.shape
    p = patch_size
    h = H // p
    w = W // p

    x = img.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, h * w, C, p, p)
    return x

def generate_discrepancy_mask(vis_img, inf_img, patch_size, mask_ratio=0.25):
    B, C, H, W = vis_img.shape
    patch_size = int(patch_size)

    p1 = patchify_focus(vis_img, patch_size)
    p2 = patchify_focus(inf_img, patch_size)

    diff = torch.abs(p1 - p2)
    score = diff.mean(dim=(2, 3, 4))
    B, N = score.shape
    N_mask = int(N * mask_ratio)

    _, ids_mask = torch.topk(score, N_mask, dim=1)

    patch_mask = torch.zeros((B, N), device=vis_img.device)
    patch_mask.scatter_(1, ids_mask, 1.0)

    h = H // patch_size
    w = W // patch_size

    patch_mask = patch_mask.view(B, h, w)

    seg_mask = patch_mask.unsqueeze(-1).unsqueeze(-1)
    seg_mask = seg_mask.repeat(1, 1, 1, patch_size, patch_size)

    seg_mask = seg_mask.permute(0, 1, 3, 2, 4)
    hard_mask = seg_mask.reshape(B, 1, H, W)

    return hard_mask

def load_mask_tensor(mask_path, device, foreground_classes=None):
    mask_img = Image.open(mask_path).convert('L')
    mask_np = np.array(mask_img)

    if foreground_classes is None:
        binary_mask = (mask_np > 0).astype(np.float32)
    else:
        binary_mask = np.zeros_like(mask_np, dtype=np.float32)
        for cls in foreground_classes:
            binary_mask[mask_np == cls] = 1.0

    binary_pil = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    return transform(binary_pil).unsqueeze(0).to(device)

def load_image_tensor(img_path, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('L')
    return transform(img).unsqueeze(0).to(device)

def restore_rgb_from_ycbcr(model_y_output, original_vis_path, target_size=(224, 224)):
    orig_rgb = Image.open(original_vis_path).convert('RGB')
    orig_rgb = orig_rgb.resize(target_size, Image.BICUBIC)

    orig_ycbcr = orig_rgb.convert('YCbCr')
    _, cb, cr = orig_ycbcr.split()

    h, w = model_y_output.shape
    if (h, w) != target_size:
        model_y_output = cv2.resize(model_y_output, target_size, interpolation=cv2.INTER_CUBIC)

    y_uint8 = np.clip(model_y_output * 255.0, 0, 255).astype(np.uint8)
    new_y = Image.fromarray(y_uint8, mode='L')

    final_ycbcr = Image.merge('YCbCr', (new_y, cb, cr))
    final_rgb = final_ycbcr.convert('RGB')

    return np.array(final_rgb)

@torch.no_grad()
def get_noise_prediction(model, x_noisy_spatial, t, cond_ll, cond_hf):
    noise_pred = model.dit_model(
        x_spatial=x_noisy_spatial,
        timestep=t,
        cond_spatial_ll=cond_ll,
        cond_spatial_hf=cond_hf
    )
    return noise_pred

@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index, cond_ll, cond_hf,
             z_base_scaled=None, mask=None, base_noise=None):
    predicted_noise = get_noise_prediction(model, x, t, cond_ll, cond_hf)

    betas_t = scheduler.extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = scheduler.extract(
        torch.sqrt(1.0 / scheduler.alphas), t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        x_next = model_mean
    else:
        posterior_variance_t = scheduler.extract(scheduler.betas, t, x.shape)
        step_noise = torch.randn_like(x)
        x_next = model_mean + torch.sqrt(posterior_variance_t) * step_noise

    if mask is not None and z_base_scaled is not None and base_noise is not None:
        if t_index == 0:
            z_ref = z_base_scaled
        else:
            t_prev = torch.full((x.shape[0],), t_index - 1,
                                device=x.device, dtype=torch.long)
            z_ref = scheduler.q_sample(z_base_scaled, t_prev, base_noise)

        x_next = mask * x_next + (1.0 - mask) * z_ref

    return x_next

@torch.no_grad()
def p_sample_loop_sdedit(model, scheduler, z_base_scaled, cond_ll, cond_hf, device, start_step=400, mask=None):
    b = z_base_scaled.shape[0]

    base_noise = torch.randn_like(z_base_scaled)

    t_start = torch.full((b,), start_step, device=device, dtype=torch.long)

    img = scheduler.q_sample(z_base_scaled, t_start, base_noise)

    print(f"Masked SDEdit Sampling Start from step {start_step}...")

    for i in tqdm(reversed(range(0, start_step)), desc='Sampling Loop', total=start_step):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(
            model, scheduler, img, t, i, cond_ll, cond_hf,
            z_base_scaled=z_base_scaled, mask=mask, base_noise=base_noise
        )

    return img

@torch.no_grad()
def p_sample_loop(model, scheduler, shape, cond, device):
    b = shape[0]
    img = torch.randn(shape, device=device)

    print("Sampling Start...")
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc='sampling loop', total=scheduler.timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, scheduler, img, t, i, cond)
    return img

@torch.no_grad()
def decode_image(model, generated_z_spatial):
    generated_z_seq = rearrange(generated_z_spatial, 'b c h w -> b (h w) c')

    z_dec = model.latent_to_dec(generated_z_seq)
    x_full = z_dec + model.decoder_pos_embed

    for blk in model.decoder_blocks:
        x_full = blk(x_full)

    x_full = model.decoder_norm(x_full)
    x_out = model.Decoder_pred(x_full)
    x_out = model.unpatchify(x_out)

    return x_out

def stage1(freq='low'):
    low_vmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage1_low_vmae.pth"
    high_vmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage1_high_vmae.pth"

    vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\vi\01016N.png"
    inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\ir\01016N.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ch = 1 if freq == 'low' else 3

    model = LDMAE(in_channels=ch, img_size=112, patch_size=4).to(device)

    if freq == 'low':
        checkpoint1 = torch.load(low_vmae_pt_path, map_location=device)

    else:
        checkpoint1 = torch.load(high_vmae_pt_path, map_location=device)

    filtered_state_dict = {k: v for k, v in checkpoint1['model_state'].items() if 'dit_model' not in k and 'cond_encoder' not in k}
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
            img1_freq = rearrange(vis_HF_list[0], 'b c d h w -> b (c d) h w')
            img2_freq = rearrange(ir_HF_list[0], 'b c d h w -> b (c d) h w')

        x, params = model(img1_freq, img2_freq, stage=1)
        posterior = DiagonalGaussianDistribution(params)
        for _ in range(20):
            samp_x, _= model(img1_freq, img2_freq, stage=1, sample_latent=True)
            samples.append(samp_x.cpu())

    if freq == 'low':
        vis = img1_freq.squeeze().detach().cpu().numpy()
        inf = img2_freq.squeeze().detach().cpu().numpy()
        sum_feature = inf + vis
        x_img = x.squeeze().detach().cpu().numpy()

        color_recon_vis = restore_rgb_from_ycbcr(vis, vis_img_path)
        color_recon_sum_feature = restore_rgb_from_ycbcr(sum_feature, vis_img_path)
        color_recon_x_img = restore_rgb_from_ycbcr(x_img, vis_img_path)

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
        ########################################
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title("Visible Input (RGB)")
        plt.imshow(color_recon_vis)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Infrared Input")
        plt.imshow(inf, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("Summation feature (RGB)")
        plt.imshow(color_recon_sum_feature)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Model Output (Recon RGB)")
        plt.imshow(color_recon_x_img)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # plt.figure(figsize=(20, 8))
        # for i, img in enumerate(samples):
        #     plt.subplot(4, 5, i + 1)
        #     plt.imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
        #     plt.title(f"sample {i + 1}")
        #     plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()

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
    low_ldmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage2_low_ldmae.pth"
    high_ldmae_pt_path = r"C:\Users\12wkd\Desktop\best_stage2_high_ldmae.pth"

    # 01016N 00040N
    vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\vi\00931N.png"
    inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\ir\00931N.png"
    # seg_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\Segmentation_labels\01016N.png"
    test_cond_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\vi\00931N.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ch = 1 if freq == 'low' else 3

    model = LDMAE(in_channels=ch, img_size=112, patch_size=4).to(device)
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='cosine')

    checkpoint_path = low_ldmae_pt_path if freq == 'low' else high_ldmae_pt_path
    checkpoint2 = torch.load(checkpoint_path, map_location=device)

    ### 지워야함###
    state_dict = checkpoint2['model_state']
    model_state = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.size() == model_state[k].size()
    }
    ################################

    model.load_state_dict(filtered_state_dict, strict=False)
    if 'scale_factor' in checkpoint2:
        model.scale_factor = torch.tensor(checkpoint2['scale_factor']).to(device)
        model.latent_mean = torch.tensor(checkpoint2.get('latent_mean', 0.0)).to(device)
    model.eval()

    vis_img = load_image_tensor(vis_img_path, device)
    inf_img = load_image_tensor(inf_img_path, device)
    test_cond_img = load_image_tensor(test_cond_path, device)
    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)

    same_cond = vis_img + inf_img
    with torch.no_grad():
        vis_LL, vis_HF_list = dwt(vis_img)
        ir_LL, ir_HF_list = dwt(inf_img)
        # test_cond_ll, test_cond_hf = dwt(test_cond_img)

        ir_ll_cond = ir_LL.squeeze(2) if ir_LL.ndim == 5 else ir_LL
        vis_hf_cond = rearrange(vis_HF_list[0], 'b c d h w -> b (c d) h w')

        # ir_ll_cond = test_cond_ll.squeeze(2) if ir_LL.ndim == 5 else ir_LL
        # vis_hf_cond = rearrange(test_cond_hf[0], 'b c d h w -> b (c d) h w')

        cond_ll = F.interpolate(ir_ll_cond, size=(112, 112), mode='bilinear')
        cond_hf = F.interpolate(vis_hf_cond, size=(112, 112), mode='bilinear')
        cond_ll_encoded = model.cond_encoder_ll(cond_ll)
        cond_hf_encoded = model.cond_encoder_hf(cond_hf)

        if freq == 'low':
            img1_freq, img2_freq = vis_LL, ir_LL
        else:
            img1_freq = rearrange(vis_HF_list[0], 'b c d h w -> b (c d) h w')
            img2_freq = rearrange(ir_HF_list[0], 'b c d h w -> b (c d) h w')

        posterior = model._encode_to_spatial_latent(img1_freq, img2_freq)
        z_base = posterior.mean

        grid = int(z_base.shape[1] ** 0.5)
        z_spatial = rearrange(z_base, 'b (h w) c -> b c h w', h=grid, w=grid)
        z_base_scaled = (z_spatial - model.latent_mean) * model.scale_factor

        # mask_tensor = load_mask_tensor(seg_path, device)



        latent_h = int(z_base.shape[1] ** 0.5)
        pixel_patch_size = vis_img.shape[-1] // latent_h\

        mask_tensor = generate_discrepancy_mask(vis_img, inf_img, patch_size= pixel_patch_size, mask_ratio=0.25)
        mask_latent = F.interpolate(mask_tensor, size=(latent_h, latent_h), mode='nearest')

        start_step = 100

        generated_z_spatial = p_sample_loop_sdedit(
            model, scheduler, z_base_scaled, cond_ll_encoded, cond_hf_encoded, device, start_step=start_step, mask=mask_latent
        )


        generated_z_spatial = (generated_z_spatial / model.scale_factor) + model.latent_mean
        output = decode_image(model, generated_z_spatial)

    if freq == 'low':
        vis_np = img1_freq.squeeze().cpu().numpy()
        inf_np = img2_freq.squeeze().cpu().numpy()
        out_low = output.squeeze().cpu().numpy()

        sum_feature = vis_np + inf_np

        color_recon_vis = restore_rgb_from_ycbcr(vis_np, vis_img_path)
        color_recon_sum_feature = restore_rgb_from_ycbcr(sum_feature, vis_img_path)
        color_recon_x_img = restore_rgb_from_ycbcr(out_low, vis_img_path)

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

        ########################################
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title("Visible Input (RGB)")
        plt.imshow(color_recon_vis)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Infrared Input")
        plt.imshow(inf_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("Summation feature (RGB)")
        plt.imshow(color_recon_sum_feature)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Model Output (Recon RGB)")
        plt.imshow(color_recon_x_img)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return out_low

    else:
        vh_np = img1_freq[0].detach().cpu().numpy()
        ih_np = img2_freq[0].detach().cpu().numpy()
        out_high = output[0].detach().cpu().numpy()

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
    # low_feat = stage1(freq='low')
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
    # # final_img = np.clip(final_img, 0, 1)
    #
    # color_recon_final_img = restore_rgb_from_ycbcr(final_img, r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\vi\01016N.png")
    # plt.figure(figsize=(6, 6))
    # plt.title("Final Output Image(IDWT)")
    # plt.imshow(color_recon_final_img)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #
    low_feat = stage2(freq='low')
    high_feat = stage2(freq='high')

    low_tensor = torch.tensor(low_feat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    high_tensor = torch.tensor(high_feat, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    ifm = DWTInverse(mode="periodization", wave='haar').to(device)

    inverse_out = ifm((low_tensor, [high_tensor]))

    final_img = inverse_out.squeeze().detach().cpu().numpy()
    # final_img = np.clip(final_img, 0, 1)

    vis_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\vi\00931N.png"
    inf_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\ir\00931N.png"

    TARGET_SIZE = (224, 224)

    orig_vis = Image.open(vis_path).convert('RGB')
    vis_np = np.array(orig_vis)
    vis_res = cv2.resize(vis_np, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    orig_inf = Image.open(inf_path).convert('L')
    inf_np = np.array(orig_inf)
    inf_res = cv2.resize(inf_np, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # 01016N 00040N 00095D 00466D
    color_recon_final_img = restore_rgb_from_ycbcr(final_img, r"C:\Users\12wkd\Desktop\experiments\MMIF\MSRS-main\test\vi\00931N.png")
    color_recon_final_img = cv2.resize(color_recon_final_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Visible Modal")
    plt.imshow(vis_res)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Original Infrared Modal")
    plt.imshow(inf_res, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Final Output Image(IDWT)")
    plt.imshow(color_recon_final_img)
    plt.axis('off')


    plt.tight_layout()
    plt.show()
