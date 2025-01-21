from elasticdino.model.layers import ResidualBlock, Activation, ProjectionLayer
import torch.nn as nn
import torch
from elasticdino.training.dpt import DPTHead

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels),
            ProjectionLayer(in_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            ResidualBlock(in_channels),
            ProjectionLayer(in_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_features, hidden_size=128):
        super(UNet, self).__init__()
        self.inc = nn.Sequential(
            ProjectionLayer(n_features, hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
        )
        self.down1 = (Down(hidden_size, hidden_size * 2))
        self.down2 = (Down(hidden_size * 2, hidden_size * 4))
        self.down3 = (Down(hidden_size * 4, hidden_size * 8))
        self.down4 = (Down(hidden_size * 8, hidden_size * 16 // 2))
        self.up1 = (Up(hidden_size * 16, hidden_size * 8 // 2))
        self.up2 = (Up(hidden_size * 8, hidden_size * 4 // 2))
        self.up3 = (Up(hidden_size * 4, hidden_size * 2 // 2))
        self.up4 = (Up(hidden_size * 2, hidden_size))
        self.outc = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, 1),
            Activation(),
            nn.Conv2d(hidden_size // 2, hidden_size // 4, 1),
            Activation(),
            nn.Conv2d(hidden_size // 4, 1, 1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class DPTDepthModel(nn.Module):
    def __init__(self, n_features, dino, target_size):
        super().__init__()
        self.dino = dino
        self.head = DPTHead(dino.feature_size, n_features)
        self.out = nn.Sequential(nn.Upsample(target_size), nn.Identity())

    
    def forward(self, x):
        print("in model", x.shape)
        x = torch.nn.functional.interpolate(x, 224, mode="bilinear")
        with torch.no_grad():
            f = self.dino.get_intermediate_features_for_tensor(x, 4)
            f = torch.stack(f).transpose(0, 1)
        return self.out(self.head(f))

class ElasticDinoDepthModel(nn.Module):
    def __init__(self, n_features, elasticdino):
        super().__init__()
        elasticdino.requires_grad_ = False
        self.elasticdino = elasticdino.eval()
        self.head = UNet(elasticdino.config["n_features_in"], n_features)
        self.out = nn.Identity()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, 224, mode="bilinear")
        with torch.no_grad():
            f = self.elasticdino(x)
        return self.out(self.head(f))