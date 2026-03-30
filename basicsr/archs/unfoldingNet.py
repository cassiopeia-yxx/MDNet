"""
@File: ProxUnfoldNet_v2.py
@Author: yx
@Date: 2024/8/13 下午7:51
@Description:
    - PGD-based Deep Unfolding Network for Joint Low-light Image Enhancement and Deblurring
    - Incorporates Retinex decomposition, blur modeling, learnable PGD, and attention-enhanced refinement
    - This version uses specialized PGD blocks with output constraints (Sigmoid/ReLU) for R, L, and I.
@Copyright: Copyright (c) 2024, All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import LayerNorm2d

# ------------------------------ Utility Blocks ------------------------------ #
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# ------------------------------ Core PGD Modules with Output Constraints ------------------------------ #

class PGDProxBlock_Base(nn.Module):
    """Base class for PGD blocks to avoid code repetition."""
    def __init__(self, channels, out_channels, n_feat, FFN_Expand=2, drop_out_rate=0.0, use_feat_out=True):
        super().__init__()
        self.use_feat_out = use_feat_out

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.ca_branch = CALayer(channels, reduction=8)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.GELU(),
        )

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            LayerNorm2d(channels // 8),
            nn.GELU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        self.concat = nn.Conv2d(channels * 2, channels, 1)
        self.ffn_pre = nn.Conv2d(channels, FFN_Expand * channels, 1)
        self.gate = SimpleGate()
        self.ffn_post = nn.Conv2d((FFN_Expand * channels) // 2, channels, 1)

        if self.use_feat_out:
            self.feat_out = conv(channels, n_feat, 1)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward_features(self, x):
        x1 = self.norm1(x)
        spatial_feat = self.spatial_branch(x1)
        spatial_map = self.spatial_interaction(spatial_feat)
        x_for_ca = x1 * spatial_map
        ca_feat = self.ca_branch(x_for_ca)
        channel_weight = self.channel_interaction(F.adaptive_avg_pool2d(ca_feat, 1))
        spatial_feat = spatial_feat * channel_weight
        fused = self.concat(torch.cat([spatial_feat, ca_feat], dim=1))
        x_mid = x + self.dropout1(fused)

        x2 = self.norm2(x_mid)
        x_ffn = self.ffn_pre(x2)
        x_ffn = self.gate(x_ffn)
        x_ffn = self.ffn_post(x_ffn)
        return self.dropout2(x_ffn) + x_mid

    def forward(self, x):
        x_ffn_res = self.forward_features(x)
        x_out = self.out(x_ffn_res)

        if self.use_feat_out:
            f_next = self.feat_out(x_ffn_res)
            return x_out, f_next
        else:
            return x_out, None

class PGDProxBlock_L(PGDProxBlock_Base):
    """PGD Block for Illumination (L) with Sigmoid activation."""
    def __init__(self, channels, out_channels, n_feat, **kwargs):
        super().__init__(channels, out_channels, n_feat, **kwargs)
        self.out = nn.Sequential(
            conv(channels, out_channels, 1),
            nn.Sigmoid()
        )

class PGDProxBlock_R(PGDProxBlock_Base):
    """PGD Block for Reflectance (R) with ReLU activation."""
    def __init__(self, channels, out_channels, n_feat, **kwargs):
        super().__init__(channels, out_channels, n_feat, **kwargs)
        self.out = nn.Sequential(
            conv(channels, out_channels, 1),
        )

class PGDProxBlock_I(PGDProxBlock_Base):
    """PGD Block for Image (I) with ReLU activation."""
    def __init__(self, channels, out_channels, n_feat, **kwargs):
        super().__init__(channels, out_channels, n_feat, **kwargs)
        self.out = nn.Sequential(
            conv(channels, out_channels, 1),
        )

class RhoPredictor(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = conv(in_c, in_c, kernel_size=1)
        self.attn = CALayer(in_c, reduction=8)
        self.out = conv(in_c, 1, kernel_size=1)
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.attn(feat)
        rho = self.out(feat)
        return rho

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size):
        super().__init__()
        self.body = nn.Sequential(
            conv(n_feats, 64, kernel_size),
            nn.PReLU(),
            conv(64, n_feats, kernel_size)
        )

    def forward(self, x):
        return x + self.body(x)

# ------------------------------ Main PGD Unfolding Network ------------------------------ #
class unfolding_net(nn.Module):
    def __init__(self, n_feat, nums_stages):
        super().__init__()
        self.nums_stages = nums_stages

        self.rho_I = nn.ModuleList([RhoPredictor(n_feat + 3) for _ in range(nums_stages)])
        self.rho_L = nn.ModuleList([RhoPredictor(n_feat + 1) for _ in range(nums_stages)])
        self.rho_R = nn.ModuleList([RhoPredictor(n_feat + 3) for _ in range(nums_stages)])
        self.lambda1 = nn.Parameter(torch.ones(nums_stages) * 0.5)

        # Use the new specialized PGD blocks
        self.R_update = nn.ModuleList([
            PGDProxBlock_R(channels=n_feat + 10, out_channels=3, n_feat=n_feat, use_feat_out=(i != nums_stages - 1)) for i in range(nums_stages)
        ])
        self.L_update = nn.ModuleList([
            PGDProxBlock_L(channels=n_feat + 8, out_channels=1, n_feat=n_feat, use_feat_out=(i != nums_stages - 1)) for i in range(nums_stages)
        ])
        self.I_update = nn.ModuleList([
            PGDProxBlock_I(channels=n_feat + 3, out_channels=3, n_feat=n_feat, use_feat_out=(i != nums_stages - 1)) for i in range(nums_stages)
        ])

        self.feat_extract_r = nn.Conv2d(3, n_feat, 3, 1, 1)
        self.feat_extract_l = nn.Conv2d(1, n_feat, 3, 1, 1)
        self.feat_extract_i = nn.Conv2d(3, n_feat, 3, 1, 1)

        self.phi = nn.ModuleList([ResBlock(conv, 3, 3) for _ in range(nums_stages)])
        self.phit = nn.ModuleList([ResBlock(conv, 3, 3) for _ in range(nums_stages)])

    def forward(self, Y):
        ListR, ListL, ListI = [], [], []
        L = torch.max(Y, 1)[0].unsqueeze(1)
        R = Y / (L + 1e-8)
        I = Y

        f_r = self.feat_extract_r(R)
        f_l = self.feat_extract_l(L)
        f_i = self.feat_extract_i(I)

        for i in range(self.nums_stages):

            I_hat = I - self.rho_I[i](torch.cat([f_i, I], dim=1)) * self.lambda1[i] * (I - R * L)
            # Direct prediction instead of residual learning
            I, f_i_new = self.I_update[i](torch.cat([f_i, I_hat], dim=1))
            if f_i_new is not None:
                f_i = f_i_new
            ListI.append(I)

            grad_L = torch.sum((self.phit[i](self.phi[i](R * L) - Y)) * R + self.lambda1[i] * (R * L - I) * R, dim=1, keepdim=True)
            L_hat = L - self.rho_L[i](torch.cat([f_l, L], dim=1)) * grad_L
            # Direct prediction instead of residual learning
            L, f_l_new = self.L_update[i](torch.cat([L, L_hat, R, I, f_l], dim=1))
            if f_l_new is not None:
                f_l = f_l_new
            ListL.append(L)

            grad_R = (self.phit[i](self.phi[i](R * L) - Y)) * L + self.lambda1[i] * (R * L - I) * L
            R_hat = R - self.rho_R[i](torch.cat([f_r, R], dim=1)) * grad_R
            # Direct prediction instead of residual learning
            R, f_r_new = self.R_update[i](torch.cat([R, R_hat, L, I, f_r], dim=1))
            if f_r_new is not None:
                f_r = f_r_new
            ListR.append(R)

        return ListL, ListR, ListI