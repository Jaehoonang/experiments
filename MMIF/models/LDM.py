import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, einsum
from einops.layers.torch import Rearrange
from functools import partial
import math
#
# def exists(x):
#     return x is not None
#
# def default(val, d):
#     if exists(val):
#         return val
#     return d() if callable(d) else d
#
# def num_to_groups(num, divisor):
#     groups = num // divisor
#     remainder = num % divisor
#     arr = [divisor] * groups
#     if remainder > 0:
#         arr.append(remainder)
#     return arr
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x
#
# def Upsample(dim, dim_out=None):
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode="nearest"),
#         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
#     )
#
# def Downsample(dim, dim_out=None):
#     # No More Strided Convolutions or Pooling
#     return nn.Sequential(
#         Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
#         nn.Conv2d(dim * 4, default(dim_out, dim), 1),
#     )
#
# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings
#
# class WeightStandardizedConv2d(nn.Conv2d):
#     def forward(self, x):
#         eps = 1e-5 if x.dtype == torch.float32 else 1e-3
#         weight = self.weight
#         mean = reduce(weight, "o ... -> o 1 1 1", "mean")
#         var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
#         normalized_weight = (weight - mean) / (var + eps).rsqrt()
#
#         return F.conv2d(
#             x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups,)
#
# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups=8):
#         super().__init__()
#         self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.act = nn.SiLU()
#
#     def forward(self, x, scale_shift=None):
#         x = self.proj(x)
#         x = self.norm(x)
#
#         if exists(scale_shift):
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift
#
#         x = self.act(x)
#         return x
#
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
#         super().__init__()
#         self.mlp = (nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(time_emb_dim, dim_out*2))
#             if exists(time_emb_dim)
#             else None)
#
#         self.block1 = Block(dim, dim_out, groups=groups)
#         self.block2 = Block(dim_out, dim_out, groups=groups)
#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
#
#     def forward(self, x, time_emb=None):
#         scale_shift = None
#         if exists(self.mlp) and exists(time_emb):
#             time_emb = self.mlp(time_emb)
#             time_emb = rearrange(time_emb, "b c -> b c 1 1")
#             scale_shift = time_emb.chunk(2, dim=1)
#
#         h = self.block1(x, scale_shift=scale_shift)
#         h = self.block2(h)
#         return h + self.res_conv(x)
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(
#             lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
#         )
#         q = q * self.scale
#
#         sim = einsum(q, k, "b h d i, b h d j -> b h i j")
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)
#
#         out = einsum(attn, v, "b h i j, b h d j -> b h i d")
#         out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
#         return self.to_out(out)
#
# class CrossAttention(nn.Module):
#     def __init__(self, dim, cond_dim, heads=4, dim_head=32):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         hidden = heads * dim_head
#
#         self.to_q = nn.Conv2d(dim, hidden, 1, bias=False)
#         self.to_k = nn.Conv2d(cond_dim, hidden, 1, bias=False)
#         self.to_v = nn.Conv2d(cond_dim, hidden, 1, bias=False)
#
#         self.to_out = nn.Conv2d(hidden, dim, 1)
#
#     def forward(self, x, cond):
#         b, c, h, w = x.shape
#
#         q = self.to_q(x)
#         k = self.to_k(cond)
#         v = self.to_v(cond)
#
#         q = rearrange(q, "b (h d) x y -> b h d (x y)", h=self.heads)
#         k = rearrange(k, "b (h d) x y -> b h d (x y)", h=self.heads)
#         v = rearrange(v, "b (h d) x y -> b h d (x y)", h=self.heads)
#
#         q = q * self.scale
#
#         sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
#         attn = sim.softmax(dim=-1)
#
#         out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
#         out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
#
#         return self.to_out(out)
#
# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#
#         self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
#                                     nn.GroupNorm(1, dim))
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(
#             lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
#         )
#
#         q = q.softmax(dim=-2)
#         k = k.softmax(dim=-1)
#
#         q = q * self.scale
#         context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
#
#         out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
#         out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
#         return self.to_out(out)
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.GroupNorm(1, dim)
#
#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)
#
# class DiffUNet(nn.Module):
#     def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1,2,3,8), channels=3, self_condition=False, resnet_block_groups=4 ):
#         super().__init__()
#
#         self.channels = channels
#         self.self_condition = self_condition
#         input_channels = channels * (2 if self_condition else 1)
#
#         init_dim = default(init_dim, dim)
#         self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
#
#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#
#         self.cond_encoder = nn.Sequential(nn.Conv2d(channels, init_dim, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(init_dim, init_dim, 3, padding=1)
#         )
#
#         block_klass = partial(ResnetBlock, groups=resnet_block_groups)
#
#         time_dim = dim * 4
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(dim),
#             nn.Linear(dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim),
#         )
#
#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)
#
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)
#
#             self.downs.append(nn.ModuleList([
#                 block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                 block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                 Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#                 Downsample(dim_in, dim_out)
#                 if not is_last
#                 else nn.Conv2d(dim_in, dim_out, 3, padding=1),
#             ]))
#
#         mid_dim = dims[-1]
#         self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.mid_attn = CrossAttention(mid_dim, init_dim)
#         self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#
#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
#             is_last = ind == (len(in_out) - 1)
#             self.ups.append(
#                 nn.ModuleList([
#                     block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#                     block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#                     Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#                     Upsample(dim_out, dim_in)
#                     if not is_last
#                     else nn.Conv2d(dim_out, dim_in, 3, padding=1),
#                 ]))
#
#         self.out_dim = default(out_dim, channels)
#         self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
#         self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
#
#     def forward(self, x ,time, cond=None, x_self_cond=None):
#         if self.self_condition:
#             x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
#             x = torch.cat((x_self_cond, x), dim=1)
#
#         x = self.init_conv(x)
#         r = x.clone()
#         t = self.time_mlp(time)
#         h = []
#
#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             h.append(x)
#
#             x = block2(x, t)
#             x = attn(x)
#             h.append(x)
#
#             x = downsample(x)
#
#         x = self.mid_block1(x, t)
#
#         cond_feat = None
#
#         if cond is not None:
#             cond_feat = self.cond_encoder(cond)
#             cond_feat = F.interpolate(cond_feat, size=x.shape[-2:], mode="bilinear")
#             x = x + self.mid_attn(x, cond_feat)
#
#         x = self.mid_block2(x, t)
#
#         for block1, block2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block1(x, t)
#
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block2(x, t)
#             x = attn(x)
#
#             x = upsample(x)
#
#         x = torch.cat((x, r), dim=1)
#         x = self.final_res_block(x, t)
#
#         return self.final_conv(x)
#
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0.0001, 0.9999)
#
# def linear_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start, beta_end, timesteps)
#
# def quadratic_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
#
# def sigmoid_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     betas = torch.linspace(-6, 6, timesteps)
#     return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
#
########################################################################################################################################################

class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps

        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule (improved DDPM)
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # q_sample을 위한 계수들
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """정방향 확산: x_0에서 x_t를 생성"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        """배치 데이터의 인덱스에 맞는 계수를 추출"""
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

        # Q, K, V Projection
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)  # 필요시 드롭아웃 추가
        )

    def forward(self, x, context=None):
        # x: (Batch, Sequence_Length, Dim) - Noisy Image Features
        # context: (Batch, Sequence_Length, Context_Dim) - Condition (Visible Image Features)

        h = self.heads

        # 1. Q, K, V 계산
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        # 2. Head 나누기 (Batch, Heads, Seq, Dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # 3. Attention Score 계산 (Scaled Dot-Product)
        # sim: (Batch, Heads, Q_seq, K_seq)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # 4. Value와 결합
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # 5. Head 합치기 및 출력
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SpatialCrossAttention(nn.Module):
    """
    CNN(이미지) feature map을 Attention이 가능한 형태로 변환해주는 래퍼 클래스
    """

    def __init__(self, channels, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)  # GroupNorm 사용 (안정성)
        self.attn = CrossAttention(channels, context_dim, heads, dim_head)
        self.proj_in = nn.Conv2d(channels, channels, 1)  # 1x1 Conv (선택사항)

    def forward(self, x, context):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)

        # (Batch, Channel, H, W) -> (Batch, H*W, Channel) 형태로 변환
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Attention 수행
        out = self.attn(x, context)

        # 다시 이미지 형태로 복원: (Batch, H*W, Channel) -> (Batch, Channel, H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        return residual + out  # Skip Connection

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cond_proj = nn.Conv2d(1, out_ch, 1)

        if up:
            # Upsampling 시에는 Skip connection 때문에 채널이 2배가 됩니다.
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
        # 1. 첫 번째 컨볼루션
        h = self.bnorm1(self.relu(self.conv1(x)))

        # 2. 타임 임베딩 변환 및 차원 확장 (view 사용)
        time_emb = self.relu(self.time_mlp(t))
        # (Batch, Channel) -> (Batch, Channel, 1, 1)
        # 특징 맵의 H, W와 더할 수 있도록 4차원으로 맞춰줍니다.
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)

        cond = F.interpolate(cond, size=h.shape[-2:], mode='bilinear', align_corners=False)
        cond_emb = self.cond_proj(cond)
        # 3. 이미지 특징과 시간 정보 결합
        h = h + time_emb + cond_emb

        # 4. 두 번째 컨볼루션 및 해상도 변경(Down/Up)
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

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim,) \
                                    for i in range(len(down_channels) - 1)])

        # Upsample
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
