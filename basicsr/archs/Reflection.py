import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from .mamba import DualSSM

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=2, stride=2, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    

##########################################################################
##---------- Restormer -----------------------
class Reflection_adjustment(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3,
        dim = 32,
        num_blocks = [1, 2, 4], 
        num_refinement_blocks = 4,
    ):

        super(Reflection_adjustment, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[DualSSM(dim=dim) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[DualSSM(dim=int(dim*2**1)) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[DualSSM(dim=int(dim*2**2)) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[DualSSM(dim=int(dim*2**1)) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1
        self.decoder_level1 = nn.Sequential(*[DualSSM(dim=int(dim*2**1)) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[DualSSM(dim=int(dim*2**1)) for i in range(num_refinement_blocks)])
        
        self.recover = nn.Sequential(nn.Conv2d(int(dim*2**1), int(dim*2**3), kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # (H, W, C)
        inp_enc_level2 = self.down1_2(out_enc_level1) # (H/2, W/2, 2C)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2) # (H/4, W/4, 4C)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level2 = self.up3_2(out_enc_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.recover(out_dec_level1)
        
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1
