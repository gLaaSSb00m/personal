<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
import logging

logging.basicConfig(level=logging.INFO)

class ViTUNet(nn.Module):
    def __init__(self):
        super(ViTUNet, self).__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.vit_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024))
        self.bottleneck = DoubleConv(2048, 1024)
        self.cbam = CBAM(1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x, verbose=False):
        if verbose:
            logging.info(f"Input shape: {x.shape}")
        x1 = self.inc(x)
        if verbose:
            logging.info(f"x1 shape (after inc): {x1.shape}")
        x2 = self.down1(x1)
        if verbose:
            logging.info(f"x2 shape (after down1): {x2.shape}")
        x3 = self.down2(x2)
        if verbose:
            logging.info(f"x3 shape (after down2): {x3.shape}")
        x4 = self.down3(x3)
        if verbose:
            logging.info(f"x4 shape (after down3): {x4.shape}")
        x5 = self.down4(x4)
        if verbose:
            logging.info(f"x5 shape (after down4): {x5.shape}")

        vit_input = F.interpolate(x, size=(224, 224), mode='bilinear')
        if verbose:
            logging.info(f"vit_input shape (after interpolate): {vit_input.shape}")
        vit_features = self.vit(vit_input)
        assert vit_features is not None, "ViT returned None"
        if verbose:
            logging.info(f"vit_features shape (from ViT): {vit_features.shape}")
        
        vit_features = self.vit_proj(vit_features)
        assert vit_features is not None, "Projection layer returned None"
        if verbose:
            logging.info(f"vit_features shape (after projection): {vit_features.shape}")

        B, C = vit_features.shape
        H, W = x5.shape[2], x5.shape[3]
        vit_features = vit_features.view(B, C, 1, 1).expand(-1, -1, H, W)
        if verbose:
            logging.info(f"vit_features shape (reshaped): {vit_features.shape}")

        assert x5.shape[1] + vit_features.shape[1] == 2048, "Shape mismatch during concatenation"
        x5 = torch.cat([x5, vit_features], dim=1)
        if verbose:
            logging.info(f"x5 shape (after concatenation): {x5.shape}")
        x5 = self.bottleneck(x5)
        if verbose:
            logging.info(f"x5 shape (after bottleneck): {x5.shape}")
        x5 = self.cbam(x5)
        if verbose:
            logging.info(f"x5 shape (after CBAM): {x5.shape}")

        x = self.up4(x5, x4)
        if verbose:
            logging.info(f"x shape (after up4): {x.shape}")
        x = self.up3(x, x3)
        if verbose:
            logging.info(f"x shape (after up3): {x.shape}")
        x = self.up2(x, x2)
        if verbose:
            logging.info(f"x shape (after up2): {x.shape}")
        x = self.up1(x, x1)
        if verbose:
            logging.info(f"x shape (after up1): {x.shape}")

        output = torch.sigmoid(self.outc(x))
        if verbose:
            logging.info(f"Output shape: {output.shape}")
        return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)
=======
# vit_unet_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTUNet(nn.Module):
    def __init__(self):
        super(ViTUNet, self).__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.vit_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024))
        self.bottleneck = DoubleConv(2048, 1024)
        self.cbam = CBAM(1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        vit_input = F.interpolate(x, size=(224, 224), mode='bilinear')
        vit_features = self.vit(vit_input)
        vit_features = self.vit_proj(vit_features)
        B, C = vit_features.shape
        H, W = x5.shape[2], x5.shape[3]
        vit_features = vit_features.view(B, C, 1, 1).expand(-1, -1, H, W)
        x5 = torch.cat([x5, vit_features], dim=1)
        x5 = self.bottleneck(x5)
        x5 = self.cbam(x5)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return torch.sigmoid(self.outc(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

# Optional test block, guarded
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTUNet().to(device)
    from torchinfo import summary
    summary(model, input_size=(1, 3, 256, 256))
>>>>>>> 457f404a934fbe131184064cf124eba682766aa8
