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
    def __init__(self, dim=64, num_heads=2, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.conv_q = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_q,stride=stride,bias=qkv_bias,groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_k = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv,stride=stride,bias=qkv_bias,groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_v= nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv,stride=stride,bias=qkv_bias,groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.linear_proj_last = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q = rearrange(self.linear_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.linear_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.linear_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        att_score = torch.einsum('bhlk, bhtk ->bhlt', [q, k]) * self.scale
        att_map = self.attn_drop(F.softmax(att_score, dim=-1))

        x = torch.matmul(att_map, v)
        batch_size, num_heads, seq_length, depth = x.size()
        x = x.view(batch_size, seq_length, num_heads * depth)

        x = self.proj_drop(self.linear_proj_last(x))

        return x

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

    def forward(self, x, h, w):
        res = x
        x = self.norm_layer(x)
        attention = self.attention(x, h, w)
        x = res + self.layer_scale(attention)
        x  = x + self.FFN(x)

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
        self.ConvEmbed1 = ConvEmbedding(patch_size=7, in_channels=1, embed_dim=64, stride=4, padding=2)
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
        x = x1 + x2
        x, h, w = self.ConvEmbed1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att1 = self.enc_cvt1(x, h, w)
        att1 = rearrange(att1, 'b (h w) c -> b c h w', h=h, w=w)

        x, h, w = self.ConvEmbed2(att1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att2 = self.enc_cvt2(x, h, w)
        att2 = rearrange(att2, 'b (h w) c -> b c h w', h=h, w=w)

        x, h, w = self.ConvEmbed3(att2)
        x = rearrange(x, 'b c h w -> b (h w) c')
        att3 = self.enc_cvt3(x, h, w)
        att3 = rearrange(att3, 'b (h w) c -> b c h w', h=h, w=w)

        # sampling
        latent = self.to_latent(att3)
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