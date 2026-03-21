from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, einsum
from einops.layers.torch import Rearrange

import math
import numpy as np

from MMIF.utils.pos_emb import get_2d_sincos_pos_embed
from MMIF.utils.misc import DiagonalGaussianDistribution

class LDMAE(nn.Module):
    def __init__(self, in_channels=3, img_size=224, emb_dim=768, patch_size=4, num_heads=12, encoder_depth=12,
                 decoder_embed_dim=512, decoder_num_head=16, latent_dim=32, decoder_depth=8, dit_depth=4, masking=True):
        super().__init__()

        self.masking = masking
        num_patches = (img_size // patch_size) ** 2
        grid_size = img_size // patch_size

        self.Patch_Posi = Patch_Posi_embedding(in_channels, img_size, emb_dim, patch_size)
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, cls_token=False)
        self.decoder_pos_embed = nn.Parameter(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
                                              requires_grad=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.Encoder_blocks = nn.ModuleList([
            ViT_block(emb_dim, num_heads) for _ in range(encoder_depth)
        ])

        self.enc_to_latent = nn.Linear(emb_dim, latent_dim * 2)
        self.latent_to_dec = nn.Linear(latent_dim, decoder_embed_dim)

        self.pos_to_latent = nn.Linear(emb_dim, latent_dim)
        nn.init.zeros_(self.pos_to_latent.weight)
        nn.init.zeros_(self.pos_to_latent.bias)

        self.decoder_blocks = nn.ModuleList([
            ViT_block(decoder_embed_dim, decoder_num_head) for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.Decoder_pred = conv_decoder_pred(decoder_embed_dim, patch_size, in_channels, pred_with_conv=False)
        self.unet_model = StandardLDMUNet(in_channels=latent_dim)
        self.cond_encoder_2d = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(emb_dim // 2, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),)

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

    def forward(self, image1, image2, timestep=None, sample_latent=True, latent_scale=1.0, stage=1, scheduler=None):
        if image1.ndim == 5:
            image1 = rearrange(image1, 'b c d h w -> b (c d) h w')
        if image2.ndim == 5:
            image2 = rearrange(image2, 'b c d h w -> b (c d) h w')

        x_sum = image1 + image2

        if x_sum.ndim == 5:
            x_sum = rearrange(x_sum, 'b c d h w -> b (c d) h w')
        x_patches = self.Patch_Posi(x_sum)

        if stage == 1:
            score = compute_focus_score(image1, image2, patch_size=self.Patch_Posi.patch_size)
            mask, ids_keep, ids_mask, ids_restore = focus_mask(score, mask_ratio=0.25)
            x_vis = apply_focus_mask(x_patches, ids_keep)

            for blk in self.Encoder_blocks:
                x_vis = blk(x_vis)

            latent_params = self.enc_to_latent(x_vis)
            posterior = DiagonalGaussianDistribution(latent_params)

            z = posterior.sample() if sample_latent else posterior.mean

            z_dec = self.latent_to_dec(z)
            B = z_dec.shape[0]
            N_mask = ids_mask.shape[1]

            if N_mask > 0:
                mask_tokens = self.mask_token.repeat(B, N_mask, 1)
                x_concat = torch.cat([z_dec, mask_tokens], dim=1)
                x_full = torch.gather(x_concat, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_concat.shape[-1]))
            else:
                x_full = z_dec

            x_full = x_full + self.decoder_pos_embed
            for blk in self.decoder_blocks:
                x_full = blk(x_full)

            x_full = self.decoder_norm(x_full)
            x_out = self.Decoder_pred(x_full)
            x_out = self.unpatchify(x_out)

            return x_out, mask, posterior

        ########################################################
        # STAGE 2: LDM 학습 #
        ########################################################
        elif stage == 2:
            x_full_vis = x_patches
            for blk in self.Encoder_blocks:
                x_full_vis = blk(x_full_vis)

            latent_params = self.enc_to_latent(x_full_vis)
            posterior = DiagonalGaussianDistribution(latent_params)

            z_full = posterior.sample() if sample_latent else posterior.mean
            B = z_full.shape[0]

            grid_size = int(z_full.shape[1] ** 0.5)
            z_spatial = rearrange(z_full, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)

            scale_factor = 0.17545
            z_spatial = z_spatial * scale_factor

            noise = torch.randn_like(z_spatial)
            if timestep is None:
                timestep = torch.randint(0, scheduler.timesteps, (B,), device=z_spatial.device).long()
            z_noisy = scheduler.q_sample(z_spatial, timestep, noise)

            cond_img = self.cond_encoder_2d(image1)
            noise_pred = self.unet_model(x=z_noisy, timestep=timestep, cond=cond_img)

            return noise, noise_pred

#############################################################################################################
# ViT based encoder
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

#############################################################################################################
# after encoding for downsmaple and for upsample
# paper original code
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        H = int(N ** 0.5)
        assert H * H == N, 'Size mismatch.'
        x = x.reshape(B, H, H, C).permute(0, 3, 1, 2)

        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)

        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        H = int(N ** 0.5)
        assert H * H == N, 'Size mismatch.'
        x = x.reshape(B, H, H, C).permute(0, 3, 1, 2)

        if x.shape[0] >= 64:
            x = x.contiguous()

        scale_factor = 2
        if x.numel() * scale_factor > pow(2, 31):
            x = x.contiguous()
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x

class MLP_dim_resize(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_dim_resize, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class conv_decoder_pred(nn.Module):
    def __init__(self, decoder_embed_dim, patch_size, in_chans, pred_with_conv=True):
        super(conv_decoder_pred, self).__init__()
        self.p = patch_size
        self.in_chas = in_chans
        self.pred_with_conv = pred_with_conv

        if self.pred_with_conv:
            print('pred only with conv instead of previous linear')
            self.conv_smoother = nn.Conv2d(decoder_embed_dim, patch_size ** 2 * in_chans, 1, stride=1, padding=0)
        else:
            print('conv on rgb')
            self.linear_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
            self.conv_smoother = nn.Conv2d(in_chans, in_chans, 3, 1, 1)

    def forward(self, x):
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        if self.pred_with_conv:
            B = x.shape[0]
            x = x.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            x = self.conv_smoother(x)
            x = x.reshape(B, -1, h * w).permute(0, 2, 1)

        else:
            x = self.linear_pred(x)

            x = x.reshape(shape=(x.shape[0], h, w, self.p, self.p, self.in_chas))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], self.in_chas, h * self.p, w * self.p))  # B 3 256 256

            x = self.conv_smoother(x)

            x = x.reshape(x.shape[0], self.in_chas, h, self.p, w, self.p)
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(x.shape[0], h * w, self.p * self.p * self.in_chas))  # B HW C

        return x

#############################################################################################################
# top 25% difference masking
def patchify_focus(img, patch_size):
    # img: (B, C, H, W)
    if img.ndim == 5:
        img = rearrange(img, 'b c d h w -> b (c d) h w')
    B, C, H, W = img.shape
    p = patch_size
    h = H // p
    w = W // p

    x = img.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)  # B h w C p p
    x = x.reshape(B, h * w, C, p, p)
    return x

def compute_focus_score(image1, image2, patch_size):
    patch_size = int(patch_size)
    p1 = patchify_focus(image1, patch_size)
    p2 = patchify_focus(image2, patch_size)

    diff = torch.abs(p1 - p2)
    score = diff.mean(dim=(2, 3, 4))  # (B, N)
    return score

def focus_mask(score, mask_ratio=0.25):
    B, N = score.shape
    N_mask = int(N * mask_ratio)

    _, ids_mask = torch.topk(score, N_mask, dim=1)
    ids_keep = torch.argsort(score, dim=1)[:, :-N_mask]

    ids_restore = torch.argsort(
        torch.cat([ids_keep, ids_mask], dim=1),
        dim=1
    )

    mask = torch.zeros(B, N, device=score.device)
    mask.scatter_(1, ids_mask, 1.0)

    return mask, ids_keep, ids_mask, ids_restore

def apply_focus_mask_keep_grid(x, mask):
    """
    x: (B, N, C)
    mask: (B, N), 1 = masked, 0 = keep
    """
    mask = mask.unsqueeze(-1)  # (B, N, 1)
    x = x * (1.0 - mask)  # masked token = 0
    return x

def apply_focus_mask(x, ids_keep):
    """
    x: (B, N, C)
    ids_keep: (B, N_keep)
    """
    B, _, C = x.shape
    x_visible = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, C)
    )
    return x_visible

def apply_focus_mask_with_token(x, mask, mask_token):
    B, N, C = x.shape
    mask = mask.unsqueeze(-1)
    return x * (1.0 - mask) + mask * mask_token

#############################################################################################################
# diffusion Module

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0.0001, 0.9999)

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

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)
        )

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SpatialCrossAttention(nn.Module):

    def __init__(self, channels, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = CrossAttention(channels, context_dim, heads, dim_head)
        self.proj_in = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        out = self.attn(x, context)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        return residual + out


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, time_emb_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, in_dim)
        self.cond_proj = nn.Linear(cond_dim, in_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, in_dim)
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        cond_emb = self.cond_proj(cond)

        h = self.norm1(x + t_emb + cond_emb)
        out = self.net(h)

        return self.norm2(x + out)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cond_proj = nn.Conv2d(1, out_ch, 1)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, cond):
        h = self.bnorm1(self.relu(self.conv1(x)))

        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)

        cond = F.interpolate(cond, size=h.shape[-2:], mode='bilinear', align_corners=False)
        cond_emb = self.cond_proj(cond)
        h = h + time_emb + cond_emb

        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 2
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim, ) \
                                    for i in range(len(down_channels) - 1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, cond):
        t = self.time_mlp(timestep)

        x = torch.cat([x, cond], dim=1)
        x = self.conv0(x)

        residuals = []
        for down in self.downs:
            x = down(x, t, cond)
            residuals.append(x)

        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t, cond)

        return self.output(x)

class SimpleUNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        self.context_dim = 128

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(down_channels) - 1):
            self.downs.append(Block(down_channels[i], down_channels[i + 1], time_emb_dim))

        self.mid_attn = SpatialCrossAttention(channels=256, context_dim=self.context_dim)

        self.ups = nn.ModuleList([])
        for i in range(len(up_channels) - 1):
            self.ups.append(Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True))

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(1, self.context_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )

    def forward(self, x, timestep, cond):
        t = self.time_mlp(timestep)

        cond_emb = self.cond_encoder(cond)
        cond_seq = rearrange(cond_emb, 'b c h w -> b (h w) c')

        x = self.conv0(x)

        residuals = []
        for down in self.downs:
            x = down(x, t, cond)
            residuals.append(x)

        x = self.mid_attn(x, cond_seq)

        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t, cond)

        return self.output(x)

class DiTBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, time_emb_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.self_att = MultiHeadSelfAtt(emb_dim, num_heads)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.cross_att = CrossAttention(query_dim=emb_dim, context_dim=cond_dim, heads=num_heads,
                                        dim_head=emb_dim // num_heads)

        self.norm3 = nn.LayerNorm(emb_dim)
        self.ffn = FFN(emb_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, emb_dim * 6)
        )

    def forward(self, x, t_emb, cond):
        scale_shift = self.adaLN_modulation(t_emb)
        scale1, shift1, scale2, shift2, scale3, shift3 = scale_shift.unsqueeze(1).chunk(6, dim=-1)

        h1 = self.norm1(x) * (1 + scale1) + shift1
        x = x + self.self_att(h1)

        h2 = self.norm2(x) * (1 + scale2) + shift2
        x = x + self.cross_att(h2, cond)

        h3 = self.norm3(x) * (1 + scale3) + shift3
        x = x + self.ffn(h3)
        return x


class LDM_SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)

        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = CrossAttention(query_dim=channels, context_dim=channels, heads=heads, dim_head=dim_head)

        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = CrossAttention(query_dim=channels, context_dim=context_dim, heads=heads, dim_head=dim_head)

        self.norm3 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = self.proj_in(x)

        x_seq = rearrange(x, 'b c h w -> b (h w) c')

        x_seq = x_seq + self.attn1(self.norm1(x_seq), self.norm1(x_seq))
        x_seq = x_seq + self.attn2(self.norm2(x_seq), context)  # 여기서 cond_seq가 주입됨!
        x_seq = x_seq + self.ffn(self.norm3(x_seq))

        x = rearrange(x_seq, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)

        return residual + x

class LDMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        conv_in_ch = 2 * in_ch if up else in_ch
        self.conv1 = nn.Conv2d(conv_in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.silu = nn.SiLU()

    def forward(self, x, t):
        h = self.bnorm1(self.silu(self.conv1(x)))

        time_emb = self.silu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        h = self.bnorm2(self.silu(self.conv2(h)))
        return self.transform(h)

class StandardLDMUNet(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()

        image_channels = in_channels
        cond_channels = in_channels

        total_in_channels = image_channels + cond_channels

        out_dim = in_channels

        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        time_emb_dim = 128
        self.context_dim = 128

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, self.context_dim, 3, padding=1),
            nn.SiLU()
        )

        self.conv0 = nn.Conv2d(total_in_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([])
        self.down_attns = nn.ModuleList([])
        for i in range(len(down_channels) - 1):
            self.downs.append(LDMBlock(down_channels[i], down_channels[i + 1], time_emb_dim))
            self.down_attns.append(LDM_SpatialTransformer(down_channels[i + 1], self.context_dim))

        self.mid_attn = LDM_SpatialTransformer(down_channels[-1], self.context_dim)

        self.ups = nn.ModuleList([])
        self.up_attns = nn.ModuleList([])
        for i in range(len(up_channels) - 1):
            self.ups.append(LDMBlock(up_channels[i], up_channels[i + 1], time_emb_dim, up=True))
            self.up_attns.append(LDM_SpatialTransformer(up_channels[i + 1], self.context_dim))

        self.output = nn.Sequential(
            nn.Conv2d(up_channels[-1] + down_channels[0], 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 1)
        )

    def forward(self, x, timestep, cond):
        t = self.time_mlp(timestep)

        cond_emb = self.cond_encoder(cond)
        cond_seq = rearrange(cond_emb, 'b c h w -> b (h w) c')

        x = torch.cat([x, cond], dim=1)
        x = self.conv0(x)

        x_skip_28 = x
        residuals = []

        for down, attn in zip(self.downs, self.down_attns):
            x = down(x, t)
            x = attn(x, cond_seq)
            residuals.append(x)

        x = self.mid_attn(x, cond_seq)

        for up, attn in zip(self.ups, self.up_attns):
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
            x = attn(x, cond_seq)

        x = torch.cat([x, x_skip_28], dim=1)

        return self.output(x)
