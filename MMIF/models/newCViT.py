import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import to_2tuple

################################################################################################################
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
################################################################################################################
# M1, M2, M1+M2, 3 parallel CVT
# M1, M2 is condition for summation of those two so output is just one feature
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
        self.conv_q_3 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                                padding=padding_q, stride=stride, bias=qkv_bias, groups=dim),
                                      nn.BatchNorm2d(dim),
                                      nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                      Rearrange('b c h w -> b (h w) c'))

        self.conv_k_3 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                                padding=padding_kv, stride=stride, bias=qkv_bias, groups=dim),
                                      nn.BatchNorm2d(dim),
                                      nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                      Rearrange('b c h w -> b (h w) c'))

        self.conv_v_3 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
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
        self.linear_q3 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k3 = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v3 = nn.Linear(dim, dim, bias=qkv_bias)

        ################################################################################################################

    def forward(self, x1, h1, w1, x2, h2, w2):
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)
        x3 = x1 + x2

        q1, q2, q3 = self.conv_q_1(x1), self.conv_q_2(x2), self.conv_q_3(x3)
        k1, k2, k3 = self.conv_k_1(x1), self.conv_k_2(x2), self.conv_k_3(x3)
        v1, v2, v3 = self.conv_v_1(x1), self.conv_v_2(x2), self.conv_v_3(x3)
        ################################################################################################################
        q1 = rearrange(self.linear_q1(q1), 'b t (h d) -> b h t d', h=self.num_heads)
        k1 = rearrange(self.linear_k1(k1), 'b t (h d) -> b h t d', h=self.num_heads)
        v1 = rearrange(self.linear_v1(v1), 'b t (h d) -> b h t d', h=self.num_heads)

        q2 = rearrange(self.linear_q2(q2), 'b t (h d) -> b h t d', h=self.num_heads)
        k2 = rearrange(self.linear_k2(k2), 'b t (h d) -> b h t d', h=self.num_heads)
        v2 = rearrange(self.linear_v2(v2), 'b t (h d) -> b h t d', h=self.num_heads)

        q3 = rearrange(self.linear_q2(q3), 'b t (h d) -> b h t d', h=self.num_heads)
        k3 = rearrange(self.linear_k2(k3), 'b t (h d) -> b h t d', h=self.num_heads)
        v3 = rearrange(self.linear_v2(v3), 'b t (h d) -> b h t d', h=self.num_heads)

        ################################################################################################################
        self_att_score1 = torch.einsum('bhlk, bhtk ->bhlt', [q1, k1]) * self.scale
        self_att_map1 = F.softmax(self_att_score1, dim=-1)

        self_att_score2 = torch.einsum('bhlk, bhtk ->bhlt', [q2, k2]) * self.scale
        self_att_map2 = F.softmax(self_att_score2, dim=-1)

        self_att_score3 = torch.einsum('bhlk, bhtk ->bhlt', [q3, k3]) * self.scale
        self_att_map3 = F.softmax(self_att_score3, dim=-1)

        x1 = torch.matmul(self_att_map1, v1)
        x2 = torch.matmul(self_att_map2, v2)
        x3 = torch.matmul(self_att_map3, v3)

        batch_size, num_heads, seq_length, depth = x1.size()
        x1 = x1.view(batch_size, seq_length, num_heads * depth)

        batch_size, num_heads, seq_length, depth = x2.size()
        x2 = x2.view(batch_size, seq_length, num_heads * depth)

        batch_size, num_heads, seq_length, depth = x3.size()
        x3 = x3.view(batch_size, seq_length, num_heads * depth)

        return x1, x2, x3

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


    def forward(self, x1, h1, w1, x2, h2, w2):
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)

        q1, q2 = self.conv_q_1(x1), self.conv_q_2(x2)
        k1, k2 = self.conv_k_1(x1), self.conv_k_2(x2)
        v1, v2 = self.conv_v_1(x1), self.conv_v_2(x2)

        q1 = rearrange(self.linear_q1(q1), 'b t (h d) -> b h t d', h=self.num_heads)
        k1 = rearrange(self.linear_k1(k1), 'b t (h d) -> b h t d', h=self.num_heads)
        v1 = rearrange(self.linear_v1(v1), 'b t (h d) -> b h t d', h=self.num_heads)

        q2 = rearrange(self.linear_q2(q2), 'b t (h d) -> b h t d', h=self.num_heads)
        k2 = rearrange(self.linear_k2(k2), 'b t (h d) -> b h t d', h=self.num_heads)
        v2 = rearrange(self.linear_v2(v2), 'b t (h d) -> b h t d', h=self.num_heads)

        cross_att_score1 = torch.einsum('bhlk, bhtk ->bhlt', [q1, k2]) * self.scale
        cross_att_map1 = F.softmin(cross_att_score1, dim=-1)

        cross_att_score2 = torch.einsum('bhlk, bhtk ->bhlt', [q2, k1]) * self.scale
        cross_att_map2 = F.softmin(cross_att_score2, dim=-1)

        x1 = torch.matmul(cross_att_map1, v2)
        x2 = torch.matmul(cross_att_map2, v1)

        x1 = x1 + q1
        x2 = x2 + q2

        batch_size, num_heads, seq_length, depth = x1.size()
        x1 = x1.view(batch_size, seq_length, num_heads * depth)

        batch_size, num_heads, seq_length, depth = x2.size()
        x2 = x2.view(batch_size, seq_length, num_heads * depth)

        return x1, x2

################################################################################################################
# decoder
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TokenDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=2, stride=2, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            up_channels = in_channels

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride)
            up_channels = in_channels // 2

        self.conv = DoubleConv(up_channels + mid_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

################################################################################################################

# total vit block flow
class CViTFlow(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, dim, num_heads):
        super().__init__()
        emb_dim = [64, 192, 384]

        self.embedding1_1 = ConvEmbedding(7, in_channels, emb_dim[0], stride=4, padding=2,  norm_layer=None)
        self.embedding1_2 = ConvEmbedding(7, in_channels, emb_dim[0], stride=4, padding=2, norm_layer=None)
        self.embedding2_1 = ConvEmbedding(3, emb_dim[0], emb_dim[1], stride=2, padding=1, norm_layer=None)
        self.embedding2_2 = ConvEmbedding(3, emb_dim[0], emb_dim[1], stride=2, padding=1, norm_layer=None)
        self.embedding3_1 = ConvEmbedding(3, emb_dim[1], emb_dim[2], stride=2, padding=1, norm_layer=None)
        self.embedding3_2 = ConvEmbedding(3, emb_dim[1], emb_dim[2], stride=2, padding=1, norm_layer=None)

        self.norm1_1 = nn.LayerNorm(emb_dim[0])
        self.norm1_2 = nn.LayerNorm(emb_dim[0])
        self.norm2_1 = nn.LayerNorm(emb_dim[1])
        self.norm2_2 = nn.LayerNorm(emb_dim[1])
        self.norm3_1 = nn.LayerNorm(emb_dim[2])
        self.norm3_2 = nn.LayerNorm(emb_dim[2])

        self.selfconv1 = SelfConvAtt(emb_dim[0], 1, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1)
        self.crossconv2 = CrossConvAtt(emb_dim[1], 3, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1,stride=1)
        self.selfconv3 = SelfConvAtt(emb_dim[2], 6, qkv_bias=False, kernel_size=3, padding_q=1, padding_kv=1, stride=1)

        self.ffn1_1, self.ffn1_2 = FFN(in_features=emb_dim[0], hidden_features=emb_dim[0]), FFN(in_features=emb_dim[0], hidden_features=emb_dim[0])
        self.ffn2_1, self.ffn2_2 = FFN(in_features=emb_dim[1], hidden_features=emb_dim[1]), FFN(in_features=emb_dim[1], hidden_features=emb_dim[1])
        self.ffn3_1, self.ffn3_2 = FFN(in_features=emb_dim[2], hidden_features=emb_dim[2]), FFN(in_features=emb_dim[2], hidden_features=emb_dim[2])


        self.decoder1_1, self.decoder1_2 = TokenDecoder(emb_dim[2], emb_dim[1], emb_dim[1]), TokenDecoder(emb_dim[2], emb_dim[1], emb_dim[1])
        self.decoder2_1, self.decoder2_2 = TokenDecoder(emb_dim[1], emb_dim[0], emb_dim[0]), TokenDecoder(emb_dim[1], emb_dim[0], emb_dim[0])
        # self.decoder3_1, self.decoder3_2 = TokenDecoder(emb_dim[0], 1, 1,  kernel_size=7, stride=4), TokenDecoder(emb_dim[0], 1, 1,  kernel_size=7, stride=4)
        self.decoder3_1, self.decoder3_2 = UpConv(emb_dim[0], emb_dim[0]//2), UpConv(emb_dim[0], emb_dim[0]//2)
        self.outconv1, self.outconv2 = OutConv(emb_dim[0]//2, 1), OutConv(emb_dim[0]//2, 1)


    def forward(self, x1, x2):
        og_x1 = x1
        og_x2 = x2

        # CViT stage 1
        x1_1, h1, w1 = self.embedding1_1(x1)
        x1_2, h2, w2 = self.embedding1_2(x2)

        x1 = rearrange(x1_1, 'b c h w -> b (h w) c')
        x2 = rearrange(x1_2, 'b c h w -> b (h w) c')
        res1, res2 = x1, x2

        x1, x2 = self.norm1_1(x1), self.norm1_2(x2)
        x1, x2 = self.selfconv1(x1, h1, w1, x2, h2, w2)
        x1, x2 = res1 + x1, res2 + x2
        x1, x2 = self.ffn1_1(x1), self.ffn1_2(x2)

        x1_1_2d = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2_1_2d = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)

        # CViT stage 2
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)

        x2_1, h1, w1 = self.embedding2_1(x1)
        x2_2, h2, w2 = self.embedding2_2(x2)

        res1 = rearrange(x2_1, 'b c h w -> b (h w) c')
        res2 = rearrange(x2_2, 'b c h w -> b (h w) c')
        x2_1 = rearrange(x2_1, 'b c h w -> b (h w) c')
        x2_2 = rearrange(x2_2, 'b c h w -> b (h w) c')

        x1, x2 = self.norm2_1(x2_1), self.norm2_2(x2_2)
        x1, x2 = self.crossconv2(x1, h1, w1, x2, h2, w2)
        x1, x2 = x1 + res1, x2 + res2
        x1, x2 = self.ffn2_1(x1), self.ffn2_2(x2)

        x1_2_2d = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2_2_2d = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)

        # CViT stage 3
        x1_2d = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2_2d = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)

        x3_1, h1, w1 = self.embedding3_1(x1_2d)
        x3_2, h2, w2 = self.embedding3_2(x2_2d)

        x3_1 = rearrange(x3_1, 'b c h w -> b (h w) c')
        x3_2 = rearrange(x3_2, 'b c h w -> b (h w) c')
        res1, res2 = x3_1, x3_2

        x1, x2 = self.norm3_1(x3_1), self.norm3_2(x3_2)

        x1, x2 = self.selfconv3(x1, h1, w1, x2, h2, w2)
        x1, x2 = x1 + res1, x2 + res2
        x1, x2 = self.ffn3_1(x1), self.ffn3_2(x2)

        # deocder
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h1, w=w1)
        x2 = rearrange(x2, 'b (h w) c -> b c h w', h=h2, w=w2)

        # x1 = self.proj3_1(x1)
        # x2 = self.proj3_2(x2)

        x1 = self.decoder1_1(x1, x1_2_2d)
        x1 = self.decoder2_1(x1, x1_1_2d)
        x1 = self.decoder3_1(x1)
        x1 = self.outconv1(x1)

        x2 = self.decoder1_2(x2, x2_2_2d)
        x2 = self.decoder2_2(x2, x2_1_2d)
        x2 = self.decoder3_2(x2)
        x2 = self.outconv2(x2)

        x1 = F.interpolate(x1, size=og_x1.shape[-2:], mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=og_x2.shape[-2:], mode="bilinear", align_corners=False)

        return x1, x2


