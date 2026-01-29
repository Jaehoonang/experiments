import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from MMIF.utils.pos_emb import get_2d_sincos_pos_embed
from MMIF.utils.misc import DiagonalGaussianDistribution

##########################   Total VMAE Flow  ####################################
# <patch-emb + posi-emb(freeze)> - <encoder(ViT based x 6 + downsample + ViT based x 6) x 12 depth> ##
################################ <(L diffusion)> #################################
##################### <decoder x 12 depth> - <unpatchify> - out ##################
##################################################################################

class VMAE(nn.Module):
    def __init__(self, in_channels=3, img_size=224, emb_dim=768, patch_size=16, num_heads=12, encoder_depth=12,
                 decoder_embed_dim=512, decoder_num_head=16, latent_dim=32, decoder_depth=8):
        super().__init__()
        encoder_latent_dim = latent_dim
        decoder_latent_dim = latent_dim
        num_patches = (img_size // patch_size) ** 2
        grid_size = img_size // patch_size

        self.Patch_Posi = Patch_Posi_embedding(in_channels, img_size, emb_dim, patch_size)
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, cls_token=False)
        self.decoder_pos_embed = nn.Parameter(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.Encoder_blocks = nn.ModuleList()
        down_idx = encoder_depth // 2
        for i in range(encoder_depth):
            self.Encoder_blocks.append(ViT_block(emb_dim, num_heads))
            # if i == down_idx - 1:
            #     self.Encoder_blocks.append(Downsample(emb_dim, emb_dim))

        self.to_latent = MLP_dim_resize(emb_dim, latent_dim*4, encoder_latent_dim*2)
        self.from_latent = MLP_dim_resize(decoder_latent_dim, latent_dim*4, emb_dim)
        self.z_to_decoder = nn.Linear(emb_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(emb_dim, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList()
        up_idx = decoder_depth - (encoder_depth // 2)

        for i in range(decoder_depth):
            self.decoder_blocks.append(ViT_block(decoder_embed_dim, decoder_num_head))
            # if i == up_idx - 1:
            #     self.decoder_blocks.append(Upsample(decoder_embed_dim, decoder_embed_dim))


        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.Decoder_pred = conv_decoder_pred(decoder_embed_dim, patch_size, in_channels, pred_with_conv=True)


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

    def forward(self,image1, image2, sample_latent: bool = True, latent_scale: float = 1.0):
        x = image1 + image2
        x = self.Patch_Posi(x)

        score = compute_focus_score(image1, image2, patch_size=self.Patch_Posi.patch_size)
        mask, ids_keep, ids_mask, ids_restore = focus_mask(score, mask_ratio=0.25)

        x = apply_focus_mask(x, ids_keep)
        for blk in self.Encoder_blocks:
            x = blk(x)

        x_vis = x
        x_pooled = x_vis.mean(dim=1)

        latent = self.to_latent(x_pooled)
        posterior = DiagonalGaussianDistribution(latent)
        # z = posterior.sample()

        if sample_latent:
            z = posterior.sample()
            if latent_scale != 1.0:
                z = posterior.mean + latent_scale * (z - posterior.mean)
        else:
            z = posterior.mean

        z_dec = self.from_latent(z)
        z_dec = self.z_to_decoder(z_dec)

        x_vis = self.decoder_embed(x_vis)

        B = x_vis.shape[0]
        mask_tokens = self.mask_token.repeat(B, ids_mask.shape[1], 1)

        x = torch.cat([x_vis, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        )

        x = x + z_dec.unsqueeze(1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.Decoder_pred(x)

        x = self.unpatchify(x)

        return x, mask, posterior


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

