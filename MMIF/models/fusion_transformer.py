import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

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
        QKV1 = rearrange(QKV1, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, QKV=3)
        queries1, keys1, values1 = QKV1[0], QKV1[1], QKV1[2]

        res2 = modal2
        QKV2 = self.weight2(self.norm2(modal2))
        QKV2 = rearrange(QKV2, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, QKV=3)
        queries2, keys2, values2 = QKV2[0], QKV2[1], QKV2[2]

        attention_score1 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys1) / self.scale
        attention_map1 = F.softmax(attention_score1, dim=-1)
        attention_map1 = self.dropout(attention_map1)

        attention_score2 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys2) / self.scale
        attention_map2 = F.softmax(attention_score2, dim=-1)
        attention_map2 = self.dropout(attention_map2)

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
        QKV1 = rearrange(QKV1, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, QKV=3)
        queries1, keys1, values1 = QKV1[0], QKV1[1], QKV1[2]

        res2 = modal2
        QKV2 = self.weight2(self.norm2(modal2))
        QKV2 = rearrange(QKV2, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, QKV=3)
        queries2, keys2, values2 = QKV2[0], QKV2[1], QKV2[2]

        attention_score1 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys1) / self.scale
        attention_map1 = F.softmax(attention_score1, dim=-1)
        attention_map1 = self.dropout(attention_map1)

        attention_score2 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys2) / self.scale
        attention_map2 = F.softmax(attention_score2, dim=-1)
        attention_map2 = self.dropout(attention_map2)

        out1 = torch.einsum('bhal, bhlv -> bhav ', attention_map1, values1)
        out1 = rearrange(out1, "b h n d -> b n (h d)")
        out1 = res1 + self.projection1(out1)

        out2 = torch.einsum('bhal, bhlv -> bhav ', attention_map2, values2)
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
# ViT total flow
class ViTFlow(nn.Module):
    def __init__(self, in_channels, img_size, emb_size, patch_size, num_heads):
        super(ViTFlow, self).__init__()
        self.embedding1 = Patch_Posi_embedding(in_channels, img_size, emb_size, patch_size)
        self.embedding2 = Patch_Posi_embedding(in_channels, img_size, emb_size, patch_size)

        self.self_att1 = SelfAtt(emb_size, num_heads)
        self.cross_att2 = CrossAtt(emb_size, num_heads)
        self.cross_att3 = CrossAtt(emb_size, num_heads)
        self.self_att4 = SelfAtt(emb_size, num_heads)

        self.ffn1_1 = ResidualAdd(FFN(emb_size))
        self.ffn1_2 = ResidualAdd(FFN(emb_size))
        self.ffn2_1 = ResidualAdd(FFN(emb_size))
        self.ffn2_2 = ResidualAdd(FFN(emb_size))
        self.ffn3_1 = ResidualAdd(FFN(emb_size))
        self.ffn3_2 = ResidualAdd(FFN(emb_size))
        self.ffn4_1 = ResidualAdd(FFN(emb_size))
        self.ffn4_2 = ResidualAdd(FFN(emb_size))

    def forward(self, modal1, modal2):
        modal1 = self.embedding1(modal1)
        modal2 = self.embedding2(modal2)

        modal1_out, modal2_out = self.self_att1(modal1, modal2)
        modal1_out, modal2_out = self.ffn1_1(modal1_out), self.ffn1_2(modal2_out)

        modal1_out, modal2_out = self.cross_att2(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn2_1(modal1_out), self.ffn2_2(modal2_out)

        modal1_out, modal2_out = self.cross_att3(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn3_1(modal1_out), self.ffn3_2(modal2_out)

        modal1_out, modal2_out = self.self_att4(modal1_out, modal2_out)
        modal1_out, modal2_out = self.ffn4_1(modal1_out), self.ffn4_2(modal2_out)

        return modal1_out, modal2_out

###############################################################################
