import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Asks: WHICH feature channels are important?
    Example: edges matter more than flat textures for detection
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Asks: WHERE in the image should we focus?
    Example: focus on the road, ignore sky/buildings
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """
    CBAM = Channel Attention + Spatial Attention combined
    First focuses on WHAT, then focuses on WHERE
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)  # step 1: what to focus on
        x = self.spatial(x)  # step 2: where to focus
        return x