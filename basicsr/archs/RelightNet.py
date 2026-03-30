import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2d(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class SGA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            bias=False
    ):
        super().__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.film_gamma = nn.Conv2d(dim, dim, 1)  # 生成缩放 γ
        self.film_beta  = nn.Conv2d(dim, dim, 1)  # 生成偏移 β
        
        # 空间位置编码模块（Depthwise + 激活 + Depthwise）
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        )

    def forward(self, x, feats):
        """
        x:     [B, C, H, W] - 输入图像特征
        feats: [B, C, H, W] - 引导图 通常为反射层 R
        """
        b, c, h, w = x.shape

        # Step 1: QKV 构建
        qkv = self.qkv_dwconv(self.qkv(x))  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)
        v_org = v.clone()  # [B, C, H, W]
        # Step 2: FiLM
        gamma = self.film_gamma(feats)  # [B,C,H,W]
        beta = self.film_beta(feats)
        v = gamma * v + beta

        # Step 3: 注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v  # [B, head, C_head, HW]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Step 4: 输出映射
        out = self.project_out(out)

        # Step 5: 加入位置编码
        pos = self.pos_emb(v_org)
        out = out + pos

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.project_in2 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.fusion = nn.Conv2d(hidden_features*2, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, input):
        x = self.project_in1(input)
        x1, x2 = x.chunk(2, dim=1)
        x1 = F.relu(self.dwconv1(x1))
        x2 = F.relu(self.dwconv2(x2))
        x12 = self.fusion(torch.cat((x1, x2), dim=1))

        x3 = self.dwconv3(self.project_in2(input))

        output = F.gelu(x3) * x12
        output = self.project_out(output)

        return output



# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim*ffn_expansion_factor)
#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SGA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim, ffn_expansion_factor=2.66, bias=False))
            ]))

    def forward(self, x, fea_R):
        """
        x: [b,c,h,w]
        fea_R: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, feats=fea_R) + x
            x = ff(x) + x
        out = x
        return out


class Illumination_adjustment(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, dim=32, level=2, num_blocks=[1, 2, 4]):
        super(Illumination_adjustment, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection 
        self.embedding_L = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.embedding_R = nn.Conv2d(3, self.dim, 3, 1, 1, bias=False)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.output = nn.Conv2d(self.dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, L, R):
        """
        L:   [b,1,h,w]         x是feature, 不是image
        R:   [b,3,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding_L(L)
        fea_R = self.embedding_R(R)

        # Encoder
        fea_encoder = []
        fea_R_list = []
        for (IGAB, FeaDownSample, RFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, fea_R)  
            fea_R_list.append(fea_R)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            fea_R = RFeaDownsample(fea_R)

        # Bottleneck
        fea = self.bottleneck(fea, fea_R)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea_R = fea_R_list[self.level-1-i]
            fea = LeWinBlcok(fea, fea_R)
 
        # Mapping
        out = self.output(fea) + L

        return out
