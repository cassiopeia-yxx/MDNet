# -*- coding: utf-8 -*-
# Auther : ML_XX
# Date : 2025/3/20 17:10
# File : mix_degrade.py
import random
import torch
from torch import nn
from .RelightNet import Illumination_adjustment
from .unfoldingNet import unfolding_net
from .Reflection import Reflection_adjustment
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ReDUNet(nn.Module):
    def __init__(self, n_feat=32, nums_stages=5):
        super(ReDUNet, self).__init__()
        self.unfolding = unfolding_net(n_feat=n_feat, nums_stages=nums_stages)
        self.relight_L = Illumination_adjustment(dim=n_feat)
        self.relight_R = Reflection_adjustment(dim=n_feat)
        

    def forward(self, lr):
        """
        Forward pass of the unified network
        :param lr: Low-light image
        :return: Final restored image + intermediate outputs
        """
        ListL_l, ListR_l, ListI_l = self.unfolding(lr)
        L_l_re = self.relight_L(ListL_l[-1], ListR_l[-1])
        R_l_re = self.relight_R(ListR_l[-1])

        restored_image = R_l_re * L_l_re 

        return restored_image
