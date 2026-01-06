import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import to_2tuple

class ConvEmbedding(nn.Module):
    def __init__(self, patch_size=8, in_channels=1, embed_dim=128, stride=8, padding=0, norm_layer=None):
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

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q = rearrange(self.linear_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.linear_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.linear_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        att_score = torch.einsum('bhlk, bhtk ->bhlt', [q, k]) * self.scale
        att_map = F.softmax(att_score, dim=-1)

        x = torch.matmul(att_map, v)
        batch_size, num_heads, seq_length, depth = x.size()
        x = x.view(batch_size, seq_length, num_heads * depth)

        return x

class SelfConvAtt(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.conv_q_1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_q, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_k_1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_v_1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        ################################################################################################################
        self.conv_q_2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_q, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_k_2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_v_2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        ################################################################################################################
        self.linear_q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v1 = nn.Linear(dim, dim, bias=qkv_bias)

        ################################################################################################################
        self.linear_q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v2 = nn.Linear(dim, dim, bias=qkv_bias)

        ################################################################################################################


    def forward(self, x1, x2):
        # x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)
        # x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h, w=w)

        q1, q2 = self.conv_q_1(x1), self.conv_q_2(x2)
        k1, k2 = self.conv_k_1(x1), self.conv_k_2(x2)
        v1, v2 = self.conv_v_1(x1), self.conv_v_2(x2)

        q1 = rearrange(self.linear_q1(q1), 'b t (h d) -> b h t d', h=self.num_heads)
        k1 = rearrange(self.linear_k1(k1), 'b t (h d) -> b h t d', h=self.num_heads)
        v1 = rearrange(self.linear_v1(v1), 'b t (h d) -> b h t d', h=self.num_heads)

        q2 = rearrange(self.linear_q2(q2), 'b t (h d) -> b h t d', h=self.num_heads)
        k2 = rearrange(self.linear_k2(k2), 'b t (h d) -> b h t d', h=self.num_heads)
        v2 = rearrange(self.linear_v2(v2), 'b t (h d) -> b h t d', h=self.num_heads)

        cross_att_score1 = torch.einsum('bhlk, bhtk ->bhlt', [q1, k1]) * self.scale
        cross_att_map1 = F.softmax(cross_att_score1, dim=-1)

        cross_att_score2 = torch.einsum('bhlk, bhtk ->bhlt', [q2, k2]) * self.scale
        cross_att_map2 = F.softmax(cross_att_score2, dim=-1)

        x1 = torch.matmul(cross_att_map1, v1)
        x2 = torch.matmul(cross_att_map2, v2)

        batch_size, num_heads, seq_length, depth = x1.size()
        x1 = x1.view(batch_size, seq_length, num_heads * depth)

        batch_size, num_heads, seq_length, depth = x2.size()
        x2 = x2.view(batch_size, seq_length, num_heads * depth)

        return x1, x2

class CrossConvAtt(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.conv_q_1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_q, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_k_1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_v_1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        ################################################################################################################
        self.conv_q_2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_q, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_k_2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        self.conv_v_2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                              padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                    Rearrange('b c h w -> b (h w) c'))

        ################################################################################################################
        self.linear_q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v1 = nn.Linear(dim, dim, bias=qkv_bias)

        ################################################################################################################
        self.linear_q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v2 = nn.Linear(dim, dim, bias=qkv_bias)


    def forward(self, x1, x2):
        # x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)
        # x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h, w=w)

        q1, q2 = self.conv_q_1(x1), self.conv_q_2(x2)
        k1, k2 = self.conv_k_1(x1), self.conv_k_2(x2)
        v1, v2 = self.conv_v_1(x1), self.conv_v_2(x2)

        q1 = rearrange(self.linear_q1(q1), 'b t (h d) -> b h t d', h=self.num_heads)
        k1 = rearrange(self.linear_k1(k1), 'b t (h d) -> b h t d', h=self.num_heads)
        v1 = rearrange(self.linear_v1(v1), 'b t (h d) -> b h t d', h=self.num_heads)

        q2 = rearrange(self.linear_q2(q2), 'b t (h d) -> b h t d', h=self.num_heads)
        k2 = rearrange(self.linear_k2(k2), 'b t (h d) -> b h t d', h=self.num_heads)
        v2 = rearrange(self.linear_v2(v2), 'b t (h d) -> b h t d', h=self.num_heads)

        cross_att_score1 = torch.einsum('bhlk, bhtk ->bhlt', [q2, k1]) * self.scale
        cross_att_map1 = F.softmax(cross_att_score1, dim=-1)

        cross_att_score2 = torch.einsum('bhlk, bhtk ->bhlt', [q1, k2]) * self.scale
        cross_att_map2 = F.softmax(cross_att_score2, dim=-1)

        x1 = torch.matmul(cross_att_map1, v1)
        x2 = torch.matmul(cross_att_map2, v2)

        batch_size, num_heads, seq_length, depth = x1.size()
        x1 = x1.view(batch_size, seq_length, num_heads * depth)

        batch_size, num_heads, seq_length, depth = x2.size()
        x2 = x2.view(batch_size, seq_length, num_heads * depth)

        return x1, x2

################################################################################################################
# decoder
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TokenDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

################################################################################################################

class CViTFlow(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, stride, padding, dim, num_heads):
        super().__init__()
        self.embedding1_1 = ConvEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer=None)
        self.embedding1_2 = ConvEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer=None)

        self.embedding2_1 = ConvEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer=None)
        self.embedding2_2 = ConvEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer=None)

        self.embedding3_1 = ConvEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer=None)
        self.embedding3_2 = ConvEmbedding(patch_size, in_channels, embed_dim, stride, padding, norm_layer=None)

        self.selfconv1 = SelfConvAtt(dim, num_heads, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1)
        self.crossconv2 = CrossConvAtt(dim, num_heads, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1,stride=1)
        self.selfconv3 = SelfConvAtt(dim, num_heads, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1)

        self.mlp1_1, self.mlp1_2 = MLP(in_features=dim, hidden_features=dim), MLP(in_features=dim, hidden_features=dim)
        self.mlp2_1, self.mlp2_2 = MLP(in_features=dim, hidden_features=dim), MLP(in_features=dim, hidden_features=dim)
        self.mlp3_1, self.mlp3_2 = MLP(in_features=dim, hidden_features=dim), MLP(in_features=dim, hidden_features=dim)

        self.decoder1_1, self.decoder1_2 = TokenDecoder(dim, dim), TokenDecoder(dim, dim)
        self.decoder2_1, self.decoder2_2 = TokenDecoder(dim, dim), TokenDecoder(dim, dim)
        self.decoder3_1, self.decoder3_2 = TokenDecoder(dim, dim), TokenDecoder(dim, dim)


    def forward(self, x1, x2):
        # CViT stage 1
        res1, res2 = x1, x2
        x1, h1, w1 = self.embedding1_1(x1)
        x2, h2, w2 = self.embedding1_2(x2)

        x1, x2 = self.selfconv1(x1, x2)
        x1, x2 = x1 + res1, x2 + res2
        x1, x2 = self.mlp1_1(x1), self.mlp1_2(x2)

        # CViT stage 2
        res1, res2 = x1, x2
        x1, x2 = self.embedding2_1(x1), self.embedding2_2(x2)
        x1, x2 = self.crossconv2(x1, x2)
        x1, x2 = x1 + res1, x2 + res2
        x1, x2 = self.mlp2_1(x1), self.mlp2_2(x2)

        # CViT stage 3
        x1, x2 = self.embedding3_1(x1), self.embedding3_2(x2)
        res1, res2 = x1, x2
        x1, x2 = self.selfconv3(x1, x2)
        x1, x2 = x1 + res1, x2 + res2
        x1, x2 = self.mlp3_1(x1), self.mlp3_2(x2)

        # deocder
        x1 = self.decoder1_1(x1)
        x1 = self.decoder2_1(x1)
        x1 = self.decoder3_1(x1)

        x2 = self.decoder1_2(x2)
        x2 = self.decoder2_2(x2)
        x2 = self.decoder3_2(x2)

        return x1, x2

