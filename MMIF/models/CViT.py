import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import to_2tuple

class ConvEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, stride, padding, norm_layer=None):
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
        return x

class AttentionConv(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5

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

    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q = rearrange(self.linear_proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.linear_proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.linear_proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        att_score = torch.einsum('bhlk, bhtk ->bhlt', [q, k]) * self.scale
        att_map = F.softmax(att_score, dim=-1)

        x = torch.matmul(att_map, v)
        batch_size, num_heads, seq_length, depth = x.size()
        x = x.view(batch_size, seq_length, num_heads * depth)

        return x
