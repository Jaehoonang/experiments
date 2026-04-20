import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

import math
import numpy as np

from MMIF.utils.pos_emb import get_2d_sincos_pos_embed
from MMIF.utils.misc import DiagonalGaussianDistribution


#################################################
# 1. ViT Based Encoder & Decoder Modules
#################################################

class Patch_Posi_embedding(nn.Module):
    def __init__(self, in_channels, img_size, emb_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.Projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_dim))
        grid_size = img_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(emb_dim, grid_size, cls_token=False)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.Projection(x)
        x = x + self.pos_embed.to(x.device)
        return x


class MultiHeadSelfAtt(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, att_drop=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(emb_dim)
        self.weight = nn.Linear(emb_dim, emb_dim * 3)
        self.dropout = nn.Dropout(att_drop)
        self.projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        res = x
        QKV = self.weight(self.norm(x))
        QKV = rearrange(QKV, "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = QKV[0], QKV[1], QKV[2]

        attention_score = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * self.scale
        attention_map = F.softmax(attention_score, dim=-1)
        attention_map = self.dropout(attention_map)

        out = torch.einsum('bhal, bhlv -> bhav ', attention_map, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = res + self.projection(out)
        return out


class FFN(nn.Module):
    def __init__(self, emb_dim, ffn_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.FFN = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(ffn_drop))

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.FFN(x) + res
        return x


class ViT_block(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(ViT_block, self).__init__()
        self.self_att = MultiHeadSelfAtt(emb_dim, num_heads)
        self.ffn = FFN(emb_dim)

    def forward(self, x):
        x = self.self_att(x)
        x = self.ffn(x)
        return x


class conv_decoder_pred(nn.Module):
    def __init__(self, decoder_embed_dim, patch_size, in_chans, pred_with_conv=True):
        super(conv_decoder_pred, self).__init__()
        self.p = patch_size
        self.in_chas = in_chans
        self.pred_with_conv = pred_with_conv

        if self.pred_with_conv:
            self.conv_smoother = nn.Conv2d(decoder_embed_dim, patch_size ** 2 * in_chans, 1, stride=1, padding=0)
        else:
            self.linear_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
            self.conv_smoother = nn.Conv2d(in_chans, in_chans, 3, 1, 1)

    def forward(self, x):
        h = w = int(x.shape[1] ** .5)
        if self.pred_with_conv:
            B = x.shape[0]
            x = x.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            x = self.conv_smoother(x)
            x = x.reshape(B, -1, h * w).permute(0, 2, 1)
        else:
            x = self.linear_pred(x)
            x = x.reshape(shape=(x.shape[0], h, w, self.p, self.p, self.in_chas))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], self.in_chas, h * self.p, w * self.p))
            x = self.conv_smoother(x)
            x = x.reshape(x.shape[0], self.in_chas, h, self.p, w, self.p)
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(x.shape[0], h * w, self.p * self.p * self.in_chas))
        return x


#################################################
# 2. Difference Focus Noise Injection
#################################################

def patchify_focus(img, patch_size):
    if img.ndim == 5:
        img = rearrange(img, 'b c d h w -> b (c d) h w')
    B, C, H, W = img.shape
    p = patch_size
    h, w = H // p, W // p
    x = img.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, h * w, C, p, p)
    return x


def compute_focus_score(image1, image2, patch_size):
    p1 = patchify_focus(image1, int(patch_size))
    p2 = patchify_focus(image2, int(patch_size))
    diff = torch.abs(p1 - p2)
    score = diff.mean(dim=(2, 3, 4))
    return score


def get_noise_indices(score, noise_ratio=0.25):
    B, N = score.shape
    N_noise = int(N * noise_ratio)

    if N_noise == 0:
        return torch.empty((B, 0), dtype=torch.int64, device=score.device)

    _, ids_noise = torch.topk(score, N_noise, dim=1)
    return ids_noise


def apply_patch_noise(x, ids_noise, noise_std=1.0):
    B, N, C = x.shape
    if ids_noise.shape[1] == 0:
        return x

    x_noisy = x.clone()
    noise = torch.randn(B, ids_noise.shape[1], C, device=x.device) * noise_std

    ids_expanded = ids_noise.unsqueeze(-1).expand(-1, -1, C)
    x_noisy.scatter_add_(1, ids_expanded, noise)

    return x_noisy


#################################################
# 3. Diffusion Modules & Dual-Condition DiT
#################################################

class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps

        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DiTBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, time_emb_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.self_att = MultiHeadSelfAtt(emb_dim, num_heads)

        self.norm2_ll = nn.LayerNorm(emb_dim)
        self.cross_att_ll = CrossAttention(emb_dim, cond_dim, num_heads, emb_dim // num_heads)

        self.norm2_hf = nn.LayerNorm(emb_dim)
        self.cross_att_hf = CrossAttention(emb_dim, cond_dim, num_heads, emb_dim // num_heads)

        self.norm3 = nn.LayerNorm(emb_dim)
        self.ffn = FFN(emb_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, emb_dim * 8)
        )

    def forward(self, x, t_emb, cond_ll, cond_hf):
        scale_shift = self.adaLN_modulation(t_emb)
        scale1, shift1, scale2_ll, shift2_ll, scale2_hf, shift2_hf, scale3, shift3 = scale_shift.unsqueeze(1).chunk(8,
                                                                                                                    dim=-1)

        h1 = self.norm1(x) * (1 + scale1) + shift1
        x = x + self.self_att(h1)

        h2_ll = self.norm2_ll(x) * (1 + scale2_ll) + shift2_ll
        x = x + self.cross_att_ll(h2_ll, cond_ll)

        h2_hf = self.norm2_hf(x) * (1 + scale2_hf) + shift2_hf
        x = x + self.cross_att_hf(h2_hf, cond_hf)

        h3 = self.norm3(x) * (1 + scale3) + shift3
        x = x + self.ffn(h3)
        return x


class StandardDiT(nn.Module):
    def __init__(self, in_channels=4, input_size=32, patch_size=2, emb_dim=256, depth=6, num_heads=8, cond_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.grid_size = input_size // patch_size
        num_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim) * 0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

        self.cond_patch_embed_ll = nn.Conv2d(cond_dim, cond_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_proj_ll = nn.Linear(cond_dim, cond_dim)
        self.cond_pos_embed_ll = nn.Parameter(torch.randn(1, num_patches, cond_dim) * 0.02)

        self.cond_patch_embed_hf = nn.Conv2d(cond_dim, cond_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_proj_hf = nn.Linear(cond_dim, cond_dim)
        self.cond_pos_embed_hf = nn.Parameter(torch.randn(1, num_patches, cond_dim) * 0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(emb_dim=emb_dim, num_heads=num_heads, time_emb_dim=emb_dim, cond_dim=cond_dim)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(emb_dim)
        self.proj_out = nn.Linear(emb_dim, patch_size * patch_size * in_channels)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def unpatchify(self, x):
        B = x.shape[0]
        p = self.patch_size
        h = w = self.grid_size
        x = x.reshape(B, h, w, p, p, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(B, self.in_channels, h * p, w * p)
        return imgs

    def forward(self, x_spatial, timestep, cond_spatial_ll, cond_spatial_hf):
        x = self.patch_embed(x_spatial)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        x_seq = x_seq + self.pos_embed

        t_emb = self.time_mlp(timestep)

        c_ll = self.cond_patch_embed_ll(cond_spatial_ll)
        cond_seq_ll = rearrange(c_ll, 'b c h w -> b (h w) c')
        cond_seq_ll = self.cond_proj_ll(cond_seq_ll) + self.cond_pos_embed_ll

        c_hf = self.cond_patch_embed_hf(cond_spatial_hf)
        cond_seq_hf = rearrange(c_hf, 'b c h w -> b (h w) c')
        cond_seq_hf = self.cond_proj_hf(cond_seq_hf) + self.cond_pos_embed_hf

        for blk in self.blocks:
            x_seq = blk(x_seq, t_emb, cond_seq_ll, cond_seq_hf)

        x_seq = self.norm_out(x_seq)
        out_seq = self.proj_out(x_seq)
        out_spatial = self.unpatchify(out_seq)

        return out_spatial


#################################################
# 4. Main Model: LDMAE (Denoising VAE + DiT)
#################################################

class LDMAE(nn.Module):
    def __init__(self, in_channels=3, img_size=224, emb_dim=768, patch_size=8, num_heads=12, encoder_depth=12,
                 decoder_embed_dim=512, decoder_num_head=16, latent_dim=4, decoder_depth=8, dit_depth=4,
                 freeze_encoder_in_stage2=True):
        super().__init__()

        self.freeze_encoder_in_stage2 = freeze_encoder_in_stage2

        self.noise_ratio = 0.25
        self.noise_std = 1.0

        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.latent_dim = latent_dim

        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.register_buffer('latent_mean', torch.tensor(0.0))

        self.Patch_Posi = Patch_Posi_embedding(in_channels, img_size, emb_dim, patch_size)
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, self.grid_size, cls_token=False)
        self.decoder_pos_embed = nn.Parameter(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
                                              requires_grad=False)

        self.Encoder_blocks = nn.ModuleList([
            ViT_block(emb_dim, num_heads) for _ in range(encoder_depth)
        ])

        self.enc_to_latent = nn.Linear(emb_dim, latent_dim * 2)
        self.latent_to_dec = nn.Linear(latent_dim, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            ViT_block(decoder_embed_dim, decoder_num_head) for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.Decoder_pred = conv_decoder_pred(decoder_embed_dim, patch_size, in_channels, pred_with_conv=False)

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        cond_ll_channels = in_channels
        cond_hf_channels = in_channels * 3

        self.cond_encoder_ll = nn.Sequential(
            nn.Conv2d(1, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, latent_dim), nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(4, latent_dim), nn.SiLU()
        )

        self.cond_encoder_hf = nn.Sequential(
            nn.Conv2d(3, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, latent_dim), nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(4, latent_dim), nn.SiLU()
        )

        self.dit_model = StandardDiT(
            in_channels=latent_dim,
            input_size=self.grid_size,
            patch_size=2,
            emb_dim=256,
            depth=dit_depth,
            num_heads=8,
            cond_dim=latent_dim
        )

    @torch.no_grad()
    def calibrate_scale_factor(self, dataloader, device, num_batches=50):
        self.eval()
        stds, means = [], []

        for i, (img1, img2) in enumerate(dataloader):
            if i >= num_batches: break
            img1, img2 = img1.to(device), img2.to(device)

            x_sum = img1 + img2
            x_patches = self.Patch_Posi(x_sum)
            for blk in self.Encoder_blocks:
                x_patches = blk(x_patches)

            latent_params = self.enc_to_latent(x_patches)
            posterior = DiagonalGaussianDistribution(latent_params)

            z = posterior.sample()
            grid = int(z.shape[1] ** 0.5)
            z_spatial = rearrange(z, 'b (h w) c -> b c h w', h=grid, w=grid)

            means.append(z.mean().item())
            stds.append(z_spatial.std().item())

        self.latent_mean.data = torch.tensor(sum(means) / len(means)).to(device)
        self.scale_factor.data = torch.tensor(1.0 / (sum(stds) / len(stds))).to(device)
        print(
            f"[Calibrated] latent_mean = {self.latent_mean.item():.5f}, scale_factor = {self.scale_factor.item():.5f}")

    def _encode_to_spatial_latent(self, image1, image2):
        x_sum = image1 + image2
        x_patches = self.Patch_Posi(x_sum)

        x_vis = x_patches
        for blk in self.Encoder_blocks:
            x_vis = blk(x_vis)

        latent_params = self.enc_to_latent(x_vis)
        posterior = DiagonalGaussianDistribution(latent_params)
        return posterior

    def unpatchify(self, x):
        p = self.Patch_Posi.patch_size
        if isinstance(p, tuple): p = p[0]
        B, N, D = x.shape
        h = w = int(N ** 0.5)
        C = D // (p * p)
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(B, C, h * p, w * p)
        return imgs

    def forward(self, image1, image2, timestep=None, sample_latent=True, stage=1, scheduler=None, dwt_fn=None):
        if image1.ndim == 5:
            image1 = rearrange(image1, 'b c d h w -> b (c d) h w')
        if image2.ndim == 5:
            image2 = rearrange(image2, 'b c d h w -> b (c d) h w')

        ########################################################
        # STAGE 1: Denoising VAE 학습
        ########################################################
        if stage == 1:
            x_sum = image1 + image2
            x_patches = self.Patch_Posi(x_sum)

            if self.training and self.noise_ratio > 0:
                score = compute_focus_score(image1, image2, self.Patch_Posi.patch_size)
                ids_noise = get_noise_indices(score, noise_ratio=self.noise_ratio)
                x_vis = apply_patch_noise(x_patches, ids_noise, noise_std=self.noise_std)
            else:
                x_vis = x_patches

            for blk in self.Encoder_blocks:
                x_vis = blk(x_vis)

            latent_params = self.enc_to_latent(x_vis)
            posterior = DiagonalGaussianDistribution(latent_params)
            z = posterior.sample() if sample_latent else posterior.mean

            z_dec = self.latent_to_dec(z)
            x_full = z_dec + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                x_full = blk(x_full)

            x_full = self.decoder_norm(x_full)
            x_out = self.Decoder_pred(x_full)
            x_out = self.unpatchify(x_out)

            return x_out, latent_params

        ########################################################
        # STAGE 2: LDM(DiT) 학습
        ########################################################
        elif stage == 2:
            if dwt_fn is None:
                raise ValueError("STAGE 2 학습을 위해서는 dwt_fn이 반드시 인자로 전달되어야 합니다.")

            posterior = self._encode_to_spatial_latent(image1, image2)
            z_full = posterior.sample() if sample_latent else posterior.mean

            B = z_full.shape[0]
            grid = int(z_full.shape[1] ** 0.5)
            z_spatial = rearrange(z_full, 'b (h w) c -> b c h w', h=grid, w=grid)

            z_scaled = (z_spatial - self.latent_mean) * self.scale_factor
            noise = torch.randn_like(z_scaled)

            if timestep is None:
                timestep = torch.randint(0, scheduler.timesteps, (B,), device=z_scaled.device).long()
            z_noisy = scheduler.q_sample(z_scaled, timestep, noise)

            vis_LL, vis_HF_list = dwt_fn(image1)
            ir_LL, ir_HF_list = dwt_fn(image2)

            ir_ll_cond = ir_LL.squeeze(2) if ir_LL.ndim == 5 else ir_LL
            vis_hf_cond = rearrange(vis_HF_list[0], 'b c d h w -> b (c d) h w')

            ir_ll_cond = F.interpolate(ir_ll_cond, size=(self.img_size, self.img_size), mode='bilinear')
            vis_hf_cond = F.interpolate(vis_hf_cond, size=(self.img_size, self.img_size), mode='bilinear')

            cond_ll_encoded = self.cond_encoder_ll(ir_ll_cond)
            cond_hf_encoded = self.cond_encoder_hf(vis_hf_cond)

            noise_pred = self.dit_model(x_spatial=z_noisy, timestep=timestep, cond_spatial_ll=cond_ll_encoded,
                                        cond_spatial_hf=cond_hf_encoded)

            return noise, noise_pred