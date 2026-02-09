# CVT stage 3 -> gaussian distributor -> smapling -> sampled latent z - > decoding CVT stage 3

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import to_2tuple
from MMIF.utils.misc import DiagonalGaussianDistribution


################################################################################################################
# CVT Part
class ConvEmbedding(nn.Module):
    def __init__(self, patch_size=7, in_channels=1, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = stride,
            padding = padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x, H, W

class PartialConvEmbedding(nn.Module):
    def __init__(self, patch_size=7, in_channels=1, embed_dim=64,
                 stride=4, padding=2, norm_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.mask_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, mask):
        valid_mask = 1.0 - mask

        x = x * valid_mask
        out = self.conv(x)

        with torch.no_grad():
            mask_sum = self.mask_conv(valid_mask)
        mask_sum = torch.clamp(mask_sum, min=1e-6)

        out = out / mask_sum
        out = self.norm(out)

        mask_ds = F.max_pool2d(
            mask,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding
        )

        _, _, H, W = out.shape
        return out, mask_ds, H, W

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.FFN = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Linear(hidden_features, out_features),
        nn.Dropout(drop))

    def forward(self, x):
        x = self.FFN(x)
        return x

class AttentionConv(nn.Module):
    def __init__(self, dim=64, num_heads=2, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1,
                 attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.conv_q = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_q, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_k = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_v = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.linear_proj_last = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, h, w, attn_mask=None):
        x_img = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        q = self.conv_q(x_img)
        k = self.conv_k(x_img)
        v = self.conv_v(x_img)

        q = rearrange(self.linear_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.linear_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.linear_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        att_score = torch.einsum('bhlk, bhtk -> bhlt', q, k) * self.scale

        if attn_mask is not None:
            attn_mask_ = attn_mask.unsqueeze(1).unsqueeze(2)
            att_score = att_score.masked_fill(attn_mask_ == 1, float('-inf'))

        att_map = self.attn_drop(F.softmax(att_score, dim=-1))

        out = torch.matmul(att_map, v)  # (B, heads, T, D)

        B, Hh, T, D = out.shape
        out = out.transpose(1, 2).reshape(B, T, Hh * D)

        out = self.proj_drop(self.linear_proj_last(out))

        if attn_mask is not None:
            keep = (attn_mask == 0).unsqueeze(-1)
            out = out * keep

        return out

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return self.gamma * x

class CVTBlock(nn.Module):
    def __init__(self, dim=64, num_heads=2, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.layer_scale = LayerScale(dim)
        self.attention = AttentionConv(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, kernel_size=kernel_size,
                                       padding_q=padding_q, padding_kv=padding_kv, stride=stride, attn_drop=attn_drop, proj_drop=proj_drop)
        self.FFN = FFN(dim)

    def forward(self, x, h, w, attn_mask=None):
        res = x
        x = self.norm_layer(x)
        x = self.attention(x, h, w, attn_mask)
        x = res + self.layer_scale(x)

        if attn_mask is not None:
            keep = (attn_mask == 0).unsqueeze(-1)
            x = x * keep

        x = x + self.FFN(x)

        if attn_mask is not None:
            x = x * keep

        return x

################################################################################################################
# VAE Part
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

class UpConvEmbedding(nn.Module):
    def __init__(self, patch_size=7, in_channels=64, out_channels=1, stride=4, padding=2, output_padding=0, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.deconv(x)
        _, _, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, H, W

################################################################################################################
# Total CVT based VMAE
class CVT_VMAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        encoder_latent_dim = latent_dim
        decoder_latent_dim = latent_dim

        # stage 1
        self.PartialConvEmbed1 = PartialConvEmbedding(patch_size=7, in_channels=1, embed_dim=64, stride=4, padding=2)
        self.enc_cvt1 = CVTBlock(dim=64, num_heads=1, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0)

        # stage 2
        self.ConvEmbed2 = ConvEmbedding(patch_size=3, in_channels=64, embed_dim=192, stride=2, padding=1)
        self.enc_cvt2 = CVTBlock(dim=192, num_heads=4, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0)

        # stage 3
        self.ConvEmbed3 = ConvEmbedding(patch_size=3, in_channels=192, embed_dim=384, stride=2, padding=1)
        self.enc_cvt3 = CVTBlock(dim=384, num_heads=8, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0)

        # self.to_latent = MLP_dim_resize(384, latent_dim * 4, encoder_latent_dim * 2)
        # self.from_latent = MLP_dim_resize(decoder_latent_dim, latent_dim * 4, 384)

        self.to_latent = nn.Conv2d(in_channels=384, out_channels=latent_dim * 2, kernel_size=1)
        self.from_latent = nn.Conv2d(in_channels=latent_dim, out_channels=384, kernel_size=1)

        # up stage 1
        self.Up1 = UpConvEmbedding(patch_size=3, in_channels=384, out_channels=192, stride=2, padding=1, output_padding=1)
        self.dec_cvt1 = CVTBlock(dim=192, num_heads=8, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0)

        # up stage 2
        self.Up2 = UpConvEmbedding(patch_size=3, in_channels=192, out_channels=64, stride=2, padding=1, output_padding=1)
        self.dec_cvt2 = CVTBlock(dim=64, num_heads=4, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0)

        # up stage 3
        self.Up3 = UpConvEmbedding(patch_size=7, in_channels=64, out_channels=1, stride=4, padding=2, output_padding=1)
        self.dec_cvt3 = CVTBlock(dim=1, num_heads=1, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0)


    def forward(self, x1, x2, sample_latent: bool = True, latent_scale: float = 1.0):
        # CVT down
        patch_size = 7
        score = compute_focus_score(x1, x2, patch_size)
        mask_patch, _, _, _ = focus_mask(score, mask_ratio=0.25)

        B, _, H_img, W_img = x1.shape
        mask_img = patch_mask_to_image(mask_patch, patch_size, H_img, W_img)

        x = x1 + x2

        x, mask1, h, w = self.PartialConvEmbed1(x, mask_img)
        x = rearrange(x, 'b c h w -> b (h w) c')
        mask1 = rearrange(mask1, 'b 1 h w -> b (h w)')
        att1 = self.enc_cvt1(x, h, w, attn_mask=mask1)
        att1 = rearrange(att1, 'b (h w) c -> b c h w', h=h, w=w)

        x, h, w = self.ConvEmbed2(att1)
        mask2 = F.max_pool2d(mask1.view(B, 1, h * 2, w * 2), kernel_size=3, stride=2, padding=1).view(B, -1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att2 = self.enc_cvt2(x, h, w, attn_mask=mask2)
        att2 = rearrange(att2, 'b (h w) c -> b c h w', h=h, w=w)

        x, h, w = self.ConvEmbed3(att2)
        mask3 = F.max_pool2d(mask2.view(B, 1, h * 2, w * 2), kernel_size=3, stride=2, padding=1).view(B, -1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att3 = self.enc_cvt3(x, h, w, attn_mask=mask3)
        att3 = rearrange(att3, 'b (h w) c -> b c h w', h=h, w=w)

        # sampling
        latent = self.to_latent(att3)
        mask3_img = mask3.view(B, 1, h, w)
        valid = 1.0 - mask3_img

        latent = latent * valid + (1.0 - valid) * latent.detach()
        # latent = rearrange(latent, 'b (h w) c -> b c h w', h=h, w=w)
        posterior = DiagonalGaussianDistribution(latent)

        if sample_latent:
            z = posterior.sample()
            if latent_scale != 1.0:
                z = posterior.mean + latent_scale * (z - posterior.mean)
        else:
            z = posterior.mean

        z_dec = self.from_latent(z)  # (B, 384)
        # z_dec = z_dec[:, None, :]  # (B, 1, 384)
        # z_dec = repeat(z_dec, 'b 1 c -> b (h w) c', h=h, w=w)
        # z_dec = rearrange(z_dec, 'b (h w) c -> b c h w', h=h, w=w)

        # CVT up
        x, h, w = self.Up1(z_dec)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att1 = self.dec_cvt1(x, h, w)
        att1 = rearrange(att1, 'b (h w) c -> b c h w', h=h, w=w)

        x, h, w = self.Up2(att1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att2 = self.dec_cvt2(x, h, w)
        att2 = rearrange(att2, 'b (h w) c -> b c h w', h=h, w=w)

        x, h, w = self.Up3(att2)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # att3 = self.dec_cvt3(x, h, w)

        return x, posterior

def patch_mask_to_spatial(mask, H, W):
    B, N = mask.shape
    assert N == H * W
    return mask.view(B, 1, H, W)

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

def patch_mask_to_image(mask_patch, patch_size, H_img, W_img):
    """
    mask_patch: (B, N)   patch-level mask
    return: (B, 1, H_img, W_img) image-level mask
    """
    B, N = mask_patch.shape
    h = H_img // patch_size
    w = W_img // patch_size

    mask = mask_patch.view(B, h, w)              # (B, h, w)
    mask = mask.unsqueeze(-1).unsqueeze(-1)      # (B, h, w, 1, 1)
    mask = mask.expand(-1, -1, -1, patch_size, patch_size)
    mask = mask.reshape(B, 1, h * patch_size, w * patch_size)
    return mask