import torch
import torch.nn as nn
import torch.nn.functional as F

class SRFilters(nn.Module):
    """
    Upsample the pixel-aligned feature 
    """
    def __init__(self, order=2, in_ch=256, out_ch=128):
        super(SRFilters, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.image_factor = [0.5**(order-i) for i in range(0, order+1)]
        self.convs = nn.ModuleList([nn.Conv2d(in_ch+3, out_ch, kernel_size=3, padding=1)] +
                    [nn.Conv2d(out_ch+3, out_ch, kernel_size=3, padding=1) for i in range(order)])

    def forward(self, feat, images):
        for i, conv in enumerate(self.convs):
            im = F.interpolate(images, scale_factor=self.image_factor[i], mode='bicubic', align_corners=True) if self.image_factor[i] is not 1 else images
            feat = F.interpolate(feat, scale_factor=2, mode='bicubic', align_corners=True) if i is not 0 else feat
            feat = torch.cat([feat, im], dim=1)
            feat = self.convs[i](feat)
        return feat