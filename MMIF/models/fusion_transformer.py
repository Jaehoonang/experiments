import torch
import torch.nn as nn
import torch.nn.functional as F

# embedding+positional -> cross-attention+self-attention -> transfomer endoer -> channel-attention+cross-attention

# 3x3conv + positional embedding + GElu

class fusion_transformer(self):
    def __init__(self, ):

        super.__init__()

    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self, embed_size=768, num_heads=3, drop_out=0.1):
        super().__init__()
        self.LN1 = nn.LayerNorm(embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads, drop_out, batch_first=True)
        self.LN2 = nn.LayerNorm(embed_size)
        self.FFN = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(drop_out)
        )
        self.Dropout(drop_out)

    def forward(self, x):
        x = self.LN1(x)
        x = x + self.attention(x)
        x = x + self.FFN(self.LN2(x))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, num_encoders=1, embed_size=768, img_size=(512, 512)):
