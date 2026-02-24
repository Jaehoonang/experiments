import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, einsum
from einops.layers.torch import Rearrange
from functools import partial

import math
import numpy as np

from MMIF.utils.pos_emb import get_2d_sincos_pos_embed
from MMIF.utils.misc import DiagonalGaussianDistribution


class LDMAE(nn.Module):
    def __init__(self, in_channels=3, img_size=224, emb_dim=768, patch_size=16, num_heads=12, encoder_depth=12,
                 decoder_embed_dim=512, decoder_num_head=16, latent_dim=32, decoder_depth=8, dit_depth=4, masking=True):
        super().__init__()

        # VMAE module
        self.masking = masking
        encoder_latent_dim = latent_dim
        decoder_latent_dim = latent_dim
        num_patches = (img_size // patch_size) ** 2
        grid_size = img_size // patch_size

        self.Patch_Posi = Patch_Posi_embedding(in_channels, img_size, emb_dim, patch_size)
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, cls_token=False)
        self.decoder_pos_embed = nn.Parameter(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0), requires_grad=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.Encoder_blocks = nn.ModuleList()
        down_idx = encoder_depth // 2

        for i in range(encoder_depth):
            self.Encoder_blocks.append(ViT_block(emb_dim, num_heads))
            # if i == down_idx - 1:
            #     self.Encoder_blocks.append(Downsample(emb_dim, emb_dim))

        self.attn_pool = nn.Linear(emb_dim, 1)

        self.to_latent = MLP_dim_resize(emb_dim, latent_dim * 4, latent_dim * 2)
        self.from_latent = MLP_dim_resize(latent_dim, latent_dim * 4, emb_dim)

        time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.z_to_dit = nn.Linear(latent_dim, emb_dim)
        self.dit_blocks = nn.ModuleList([
            DiTBlock(emb_dim=emb_dim, num_heads=num_heads, time_emb_dim=time_emb_dim, cond_dim=emb_dim)
            for _ in range(dit_depth)
        ])

        self.z_to_decoder = nn.Linear(emb_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(emb_dim, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList()
        up_idx = decoder_depth - (encoder_depth // 2)

        for i in range(decoder_depth):
            self.decoder_blocks.append(ViT_block(decoder_embed_dim, decoder_num_head))

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.Decoder_pred = conv_decoder_pred(decoder_embed_dim, patch_size, in_channels, pred_with_conv=True)
        self.dit_to_noise = nn.Linear(emb_dim, latent_dim)

    def restore_with_mask_tokens(self, x_vis, ids_restore):
        B, N_vis, C = x_vis.shape
        N = ids_restore.shape[1]

        N_mask = N - N_vis
        mask_tokens = self.mask_token.repeat(B, N_mask, 1)

        x_ = torch.cat([x_vis, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )
        return x_

    def unpatchify(self, x):
        p = self.Patch_Posi.patch_size
        if isinstance(p, tuple):
            p = p[0]

        B, N, D = x.shape
        h = w = int(N ** 0.5)
        assert h * w == N, f"N={N} is not a square"

        assert D % (p * p) == 0, \
            f"Decoder_pred dim {D} not divisible by p*p={p * p}"

        C = D // (p * p)

        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(B, C, h * p, w * p)

        return imgs

    def forward(self, image1, image2, timestep=None, sample_latent=True, latent_scale=1.0, stage=1, scheduler=None):
        x = torch.abs(image1) + torch.abs(image2)
        x = self.Patch_Posi(x)

        score = compute_focus_score(image1, image2, patch_size=self.Patch_Posi.patch_size)
        mask, ids_keep, ids_mask, ids_restore = focus_mask(score, mask_ratio=0.3)

        x_vis = apply_focus_mask(x, ids_keep)
        for blk in self.Encoder_blocks:
            x_vis = blk(x_vis)

        ######################
        weights = self.attn_pool(x_vis)
        weights = torch.softmax(weights, dim=1)
        x_pooled = (x_vis * weights).sum(dim=1)

        latent = self.to_latent(x_pooled)
        posterior = DiagonalGaussianDistribution(latent)

        if sample_latent:
            z = posterior.sample()
            if latent_scale != 1.0:
                z = posterior.mean + latent_scale * (z - posterior.mean)
        else:
            z = posterior.mean

        ########################################################
        # STAGE 1 #
        if stage == 1:
            z_dec = self.from_latent(z)
            z_token = self.z_to_decoder(z_dec).unsqueeze(1)  # [B, 1, D]

            x_vis_emb = self.decoder_embed(x_vis)

            B = x_vis_emb.shape[0]
            mask_tokens = self.mask_token.repeat(B, ids_mask.shape[1], 1)

            x_concat = torch.cat([x_vis_emb, z_token, mask_tokens], dim=1)

            z_tok = x_concat[:, x_vis_emb.shape[1]:x_vis_emb.shape[1] + 1]
            x_wo_z = torch.cat([x_concat[:, :x_vis_emb.shape[1]], x_concat[:, x_vis_emb.shape[1] + 1:]], dim=1)

            x_wo_z = torch.gather(
                x_wo_z,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, x_wo_z.shape[-1])
            )

            x_dec = torch.cat([z_tok, x_wo_z], dim=1)

            z_tok = x_dec[:, :1, :]
            patch_tok = x_dec[:, 1:, :]

            patch_tok = patch_tok + self.decoder_pos_embed

            x_dec = torch.cat([z_tok, patch_tok], dim=1)

            for blk in self.decoder_blocks:
                x_dec = blk(x_dec)

            x_dec = self.decoder_norm(x_dec)
            x_dec = x_dec[:, 1:, :]
            x_out = self.Decoder_pred(x_dec)
            x_out = self.unpatchify(x_out)

            return x_out, mask, posterior

        elif stage == 2:
            B = z.shape[0]

            cond_patches = self.Patch_Posi(image1)
            cond_vis = apply_focus_mask(cond_patches, ids_keep)
            for blk in self.Encoder_blocks:
                cond_vis = blk(cond_vis)
            cond_feat = cond_vis

            noise = torch.randn_like(z)
            if timestep is None:
                timestep = torch.randint(0, scheduler.timesteps, (B,), device=z.device).long()

            z_noisy = scheduler.q_sample(z, timestep, noise)

            t_emb = self.time_mlp(timestep).to(z.device)
            z_diff = self.z_to_dit(z_noisy).unsqueeze(1)

            for dit_blk in self.dit_blocks:
                z_diff = dit_blk(z_diff, t_emb, cond=cond_feat)

            z_diff = z_diff.squeeze(1)
            noise_pred = self.dit_to_noise(z_diff)

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

        attention_score = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) / self.scale
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

    def forward(self,x):
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
            # padding = (0, 1, 0, 1)  # Pad 1 on the right (W) and 1 on the bottom (H)
            # Apply padding
            # x = F.pad(x, padding, mode='constant', value=0)
            x = self.conv_smoother(x)  # B C H W
            x = x.reshape(B, -1, h * w).permute(0, 2, 1)  # B HW C

        else:
            x = self.linear_pred(x)  # B HW p_size*p_size*3
            x = x.reshape(shape=(x.shape[0], h, w, self.p, self.p, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(shape=(x.shape[0], 3, h * self.p, w * self.p))  # B 3 256 256

            x = self.conv_smoother(x)
            x = x.reshape(x.shape[0], 3, h, self.p, w, self.p)
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(x.shape[0], h * w, self.p * self.p * 3))  # B HW C

        return x

#############################################################################################################
# top 25% difference masking
def patchify_focus(img, patch_size):
    # img: (B, C, H, W)
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
    mask = mask.unsqueeze(-1)          # (B, N, 1)
    x = x * (1.0 - mask)               # masked token = 0
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

        # 입력 + 시간 임베딩 + 조건 임베딩
        h = self.norm1(x + t_emb + cond_emb)
        out = self.net(h)

        # Skip Connection
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
        image_channels=2
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

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim,) \
                                    for i in range(len(down_channels) - 1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, cond):
        t = self.time_mlp(timestep)

        x = torch.cat([x, cond], dim=1)
        x = self.conv0(x)

        # Residual connections storage
        residuals = []
        for down in self.downs:
            x = down(x, t, cond)
            residuals.append(x)

        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)  # Skip connection
            x = up(x, t, cond)

        return self.output(x)

class SimpleUNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1  # Grayscale 가정
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # [중요] Condition(Visible) 이미지의 차원
        # Visible 이미지를 인코딩한 Feature의 채널 수 (여기선 간단히 이미지를 Flatten해서 쓴다고 가정하거나 별도 인코더 사용)
        self.context_dim = 128

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample Blocks
        self.downs = nn.ModuleList([])
        for i in range(len(down_channels) - 1):
            self.downs.append(Block(down_channels[i], down_channels[i + 1], time_emb_dim))

        # [NEW] Cross Attention Layer (Bottleneck에 추가)
        # 가장 깊은 곳(channel=256)에서 Attention 수행
        self.mid_attn = SpatialCrossAttention(channels=256, context_dim=self.context_dim)

        # Upsample Blocks
        self.ups = nn.ModuleList([])
        for i in range(len(up_channels) - 1):
            self.ups.append(Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True))

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

        # [NEW] Condition Encoder (Visible 이미지를 feature로 변환)
        # 단순함을 위해 1x1 conv로 차원만 맞춤 (더 복잡한 ViT나 CNN 사용 가능)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(1, self.context_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))  # (Batch, 128, 16, 16)으로 크기 고정
        )

    def forward(self, x, timestep, cond):
        # 1. Time Embedding
        t = self.time_mlp(timestep)

        # 2. Condition Encoding (Visible Image -> Context Sequence)
        # cond: (Batch, 1, H, W) -> encoder -> (Batch, 128, 16, 16)
        cond_emb = self.cond_encoder(cond)
        # Attention에 넣기 위해 Sequence 형태로 변환: (Batch, 16*16, 128)
        cond_seq = rearrange(cond_emb, 'b c h w -> b (h w) c')

        # 3. Initial Conv
        x = self.conv0(x)

        # 4. Downsampling
        residuals = []
        for down in self.downs:
            x = down(x, t, cond)  # 기존 Block의 cond는 단순 더하기용 (유지하거나 제거 가능)
            residuals.append(x)

        # 5. [NEW] Cross Attention (정보 주입의 핵심)
        # x는 현재 노이즈 이미지 특징, cond_seq는 Visible 이미지 특징
        x = self.mid_attn(x, cond_seq)

        # 6. Upsampling
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
            nn.Linear(time_emb_dim, emb_dim * 2)
        )

    def forward(self, x, t_emb, cond):
        scale_shift = self.adaLN_modulation(t_emb)
        scale, shift = scale_shift.unsqueeze(1).chunk(2, dim=-1)

        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.self_att(h)

        x = x + self.cross_att(self.norm2(x), cond)

        x = x + self.ffn(self.norm3(x))
        return x