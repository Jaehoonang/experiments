import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from MMIF.utils.cbam import CBAM
import math
###############################################################################
# embedding module
class Patch_Posi_embedding(nn.Module):
    def __init__(self, in_channels, img_size, emb_size, patch_size):
        super().__init__()
        self.Projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.Projection(x)
        x += self.positions

        return x

###############################################################################
# Vit inner modules
class SelfAtt(nn.Module):
    def __init__(self, emb_size, num_heads, att_drop=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(emb_size)
        self.weight1 = nn.Linear(emb_size, emb_size * 3)
        self.dropout1 = nn.Dropout(att_drop)
        self.projection1 = nn.Linear(emb_size, emb_size)

        self.norm2 = nn.LayerNorm(emb_size)
        self.weight2 = nn.Linear(emb_size, emb_size * 3)
        self.dropout2 = nn.Dropout(att_drop)
        self.projection2 = nn.Linear(emb_size, emb_size)

    def forward(self, modal1, modal2):
        # each modal ViT layer 원빵에 진행
        res1 = modal1
        QKV1 = self.weight1(self.norm1(modal1))
        QKV1 = rearrange(QKV1, "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries1, keys1, values1 = QKV1[0], QKV1[1], QKV1[2]

        res2 = modal2
        QKV2 = self.weight2(self.norm2(modal2))
        QKV2 = rearrange(QKV2, "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries2, keys2, values2 = QKV2[0], QKV2[1], QKV2[2]

        attention_score1 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys1) / self.scale
        attention_map1 = F.softmax(attention_score1, dim=-1)
        attention_map1 = self.dropout1(attention_map1)

        attention_score2 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys2) / self.scale
        attention_map2 = F.softmax(attention_score2, dim=-1)
        attention_map2 = self.dropout2(attention_map2)

        out1 = torch.einsum('bhal, bhlv -> bhav ', attention_map1, values1)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out1 = res1 + self.projection1(out1)

        out2 = torch.einsum('bhal, bhlv -> bhav ', attention_map2, values2)
        out2 = rearrange(out2, "b h n d -> b n (h d)")
        out2 = res2 + self.projection2(out2)

        return out1, out2

class CrossAtt(nn.Module):
    def __init__(self, emb_size, num_heads, att_drop=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(emb_size)
        self.weight1 = nn.Linear(emb_size, emb_size * 3)
        self.dropout1 = nn.Dropout(att_drop)
        self.projection1 = nn.Linear(emb_size, emb_size)

        self.norm2 = nn.LayerNorm(emb_size)
        self.weight2 = nn.Linear(emb_size, emb_size * 3)
        self.dropout2 = nn.Dropout(att_drop)
        self.projection2 = nn.Linear(emb_size, emb_size)


    def forward(self, modal1, modal2):
        # each modal ViT layer 원빵에 진행
        res1 = modal1
        QKV1 = self.weight1(self.norm1(modal1))
        QKV1 = rearrange(QKV1, "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries1, keys1, values1 = QKV1[0], QKV1[1], QKV1[2]

        res2 = modal2
        QKV2 = self.weight2(self.norm2(modal2))
        QKV2 = rearrange(QKV2, "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries2, keys2, values2 = QKV2[0], QKV2[1], QKV2[2]

        attention_score1 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys2) / self.scale
        # attention_score11 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys1) / self.scale
        attention_score1 = attention_score1
        attention_map1 = F.softmax(attention_score1, dim=-1)
        attention_map1 = self.dropout1(attention_map1)

        attention_score2 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys1) / self.scale
        # attention_score22 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys2) / self.scale
        attention_score2 = attention_score2
        attention_map2 = F.softmax(attention_score2, dim=-1)
        attention_map2 = self.dropout2(attention_map2)
        # softmin 해보기!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        out1 = torch.einsum('bhal, bhlv -> bhav ', attention_map1, values2)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out1 = res1 + self.projection1(out1)

        out2 = torch.einsum('bhal, bhlv -> bhav ', attention_map2, values1)
        out2 = rearrange(out2, "b h n d -> b n (h d)")
        out2 = res2 + self.projection2(out2)

        return out1, out2

class FFN(nn.Module):
    def __init__(self, emb_size, ffn_drop=0.1):
        super().__init__()
        self.FFN = nn.Sequential(
        nn.LayerNorm(emb_size),
        nn.Linear(emb_size, 4 * emb_size),
        nn.GELU(),
        nn.Dropout(ffn_drop),
        nn.Linear(4 * emb_size, emb_size),
        nn.Dropout(ffn_drop))

    def forward(self, x):
        x = self.FFN(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

###############################################################################
# # fusion module
class LinearDecoder(nn.Module):
    def __init__(self, emb_size, out_channels, patch_size, img_size):
        super().__init__()
        self.lin_deco = nn.Linear(emb_size, patch_size * patch_size * out_channels)
        self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=img_size // patch_size,
            w=img_size // patch_size, p1=patch_size, p2=patch_size, c=out_channels)


    def forward(self, x):
        x = self.lin_deco(x)
        x = self.rearrange(x)

        return x

class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, modal1, modal2):
        pass

# class TokenFusion(nn.Module):
#     def __init__(self, emb_size):
#         super().__init__()
#         self.gate = nn.Sequential(nn.Linear(emb_size * 2, emb_size),
#                                   nn.Sigmoid())
#
#     def forward(self, t1, t2):
#         gate = self.gate(torch.cat([t1, t2], dim=-1))
#         fused = gate * t1 + (1 - gate) * t2
#         return fused
#
# class Image_decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(256, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(32, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.decoder(x)
#
# class PatchDeTokenizer(nn.Module):
#     def __init__(self, emb_size, out_channels, patch_size):
#         super().__init__()
#         self.depatch = nn.ConvTranspose2d(
#             emb_size,
#             out_channels,
#             kernel_size=patch_size,
#             stride=patch_size
#         )
#
#     def forward(self, x):
#         # x: [B, 196, emb]
#         B, N, D = x.shape
#         H = W = int(N ** 0.5)
#         x = x.transpose(1, 2).view(B, D, H, W)
#         x = self.depatch(x)  # [B, out_channels, 224, 224]
#         return x
#
# class Fusion_block(nn.Module):
#     def __init__(self, emb_size, patch_size):
#         super().__init__()
#         self.tokenfusion = TokenFusion(emb_size)
#         self.detoken = PatchDeTokenizer(emb_size=emb_size,out_channels=64,patch_size=patch_size)
#
#         self.refine = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 1, 3, padding=1),
#             nn.Sigmoid())
#
#         # self.token_proj = nn.Linear(emb_size, 256)
#         # # self.cbam = CBAM(gate_channels=256)
#         # self.decoder = Image_decoder()
#
#
#     def forward(self, modal1_out, modal2_out):
#         fused_token = self.tokenfusion(modal1_out, modal2_out)
#         fused_img = self.detoken(fused_token)
#         output = self.refine(fused_img)
#         # fused_token = self.token_proj(fused_token)
#         # fused = token_to_feature(fused_token)
#         # fused = self.cbam(fused)
#         # output = self.decoder(fused)
#         output = torch.clamp(output, 0, 1)
#
#         return output
#
# def token_to_feature(x):
#     # x: [B, 196, 768]
#     B, N, D = x.shape
#     H = W = int(math.sqrt(N))  # 14
#     x = x.transpose(1, 2)  # [B, 768, 196]
#     x = x.view(B, D, H, W)  # [B, 768, 14, 14]
#     return x

###############################################################################
# ViT total flow
class ViTFlow(nn.Module):
    def __init__(self, in_channels, img_size, emb_size, patch_size, num_heads, out_channels):
        super(ViTFlow, self).__init__()
        self.embedding1 = Patch_Posi_embedding(in_channels, img_size, emb_size, patch_size)
        self.embedding2 = Patch_Posi_embedding(in_channels, img_size, emb_size, patch_size)

        self.self_att1 = SelfAtt(emb_size, num_heads)
        self.self_att2 = SelfAtt(emb_size, num_heads)
        self.cross_att3 = CrossAtt(emb_size, num_heads)
        self.cross_att4 = CrossAtt(emb_size, num_heads)
        self.self_att5 = SelfAtt(emb_size, num_heads)
        self.self_att6 = SelfAtt(emb_size, num_heads)

        self.ffn1_1 = ResidualAdd(FFN(emb_size))
        self.ffn1_2 = ResidualAdd(FFN(emb_size))
        self.ffn2_1 = ResidualAdd(FFN(emb_size))
        self.ffn2_2 = ResidualAdd(FFN(emb_size))
        self.ffn3_1 = ResidualAdd(FFN(emb_size))
        self.ffn3_2 = ResidualAdd(FFN(emb_size))
        self.ffn4_1 = ResidualAdd(FFN(emb_size))
        self.ffn4_2 = ResidualAdd(FFN(emb_size))
        self.ffn5_1 = ResidualAdd(FFN(emb_size))
        self.ffn5_2 = ResidualAdd(FFN(emb_size))
        self.ffn6_1 = ResidualAdd(FFN(emb_size))
        self.ffn6_2 = ResidualAdd(FFN(emb_size))

        self.lin_deco1 = LinearDecoder(emb_size, out_channels, patch_size, img_size)
        self.lin_deco2 = LinearDecoder(emb_size, out_channels, patch_size, img_size)

        # self.fusion = Fusion_block(emb_size, patch_size)

    def forward(self, modal1, modal2):
        modal1 = self.embedding1(modal1)
        modal2 = self.embedding2(modal2)

        modal1_out, modal2_out = self.self_att1(modal1, modal2)
        modal1_out, modal2_out = self.ffn1_1(modal1_out), self.ffn1_2(modal2_out)

        modal1_out, modal2_out = self.self_att2(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn2_1(modal1_out), self.ffn2_2(modal2_out)

        modal1_out, modal2_out = self.cross_att3(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn3_1(modal1_out), self.ffn3_2(modal2_out)

        modal1_out, modal2_out = self.cross_att4(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn4_1(modal1_out), self.ffn4_2(modal2_out)

        modal1_out, modal2_out = self.self_att5(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn5_1(modal1_out), self.ffn5_2(modal2_out)

        modal1_out, modal2_out = self.self_att6(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn6_1(modal1_out), self.ffn6_2(modal2_out)

        # modal1_out, modal2_out = self.self_att1(modal1_out, modal2_out)
        # modal1_out, modal2_out = self.ffn1_1(modal1_out), self.ffn1_2(modal2_out)

        # modal1_out, modal2_out = self.cross_att2(modal1_out, modal2_out)
        # modal1_out, modal2_out = self.ffn2_1(modal1_out), self.ffn2_2(modal2_out)
        #
        # modal1_out, modal2_out = self.self_att5(modal1_out, modal2_out)
        # modal1_out, modal2_out = self.ffn3_1(modal1_out), self.ffn3_2(modal2_out)

        # modal1_out, modal2_out = self.cross_att4(modal1_out, modal2_out)
        # modal1_out, modal2_out = self.ffn4_1(modal1_out), self.ffn4_2(modal2_out)

        # modal1_out, modal2_out = self.self_att5(modal1_out, modal2_out)
        # modal1_out, modal2_out = self.ffn5_1(modal1_out), self.ffn5_2(modal2_out)
        #
        # modal1_out, modal2_out = self.self_att6(modal1_out, modal2_out)
        # modal1_out, modal2_out = self.ffn6_1(modal1_out), self.ffn6_2(modal2_out)

        modal1_out = self.lin_deco1(modal1_out)
        modal2_out = self.lin_deco2(modal2_out)

        # output = self.fusion(modal1_out, modal2_out)

        return modal1_out, modal2_out

###############################################################################
