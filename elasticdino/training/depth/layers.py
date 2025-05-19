from elasticdino.model.layers import ResidualBlock, Activation, ProjectionLayer, DepthwiseConvolution
import torch.nn as nn
import torch
from elasticdino.training.dpt import DPTHead

import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseConvolution(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseConvolution(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # DepthwiseConvolution(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=1, blur:bool=False, leaky:float=None):
        super().__init__()
        #nf = ifnone(nf, ni)
        self.conv = nn.Conv2d(ni, nf*(scale**2), 1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = PixelShuffle_ICNR(ni=in_channels, nf=out_channels, scale=2,leaky=False,blur=False)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = x2 - x1
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MyDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ProjectionLayer(in_channels, out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            # ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class MyUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        self.up = PixelShuffle_ICNR(ni= in_channels, nf=out_channels, scale=2,leaky=False,blur=False)
        self.conv = nn.Sequential(
            ProjectionLayer(out_channels + in_channels, out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            # ResidualBlock(out_channels),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = x2 - x1
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MyUNet(nn.Module):
    def __init__(self, n_channels, n_features, n_features_out=1, bilinear=False, positive=True):
        super().__init__()
        self.bilinear = bilinear

        self.inc = nn.Sequential(
            ProjectionLayer(n_channels, n_features),
            ResidualBlock(n_features),
            ResidualBlock(n_features),
            ResidualBlock(n_features),
        )
        self.down1 = (MyDown(n_features, n_features))
        self.down2 = (MyDown(n_features, n_features))
        self.down3 = (MyDown(n_features, n_features))
        factor = 2 if bilinear else 1
        self.down4 = (MyDown(n_features, n_features))
        self.up1 = (MyUp(n_features, n_features, bilinear))
        self.up2 = (MyUp(n_features, n_features, bilinear))
        self.up3 = (MyUp(n_features, n_features, bilinear))
        self.up4 = (MyUp(n_features, n_features, bilinear))
        self.outc = nn.Sequential(
            DepthwiseConvolution(n_features, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            DepthwiseConvolution(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            DepthwiseConvolution(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            DepthwiseConvolution(16, n_features_out, kernel_size=1, stride=1, padding=0),
            nn.Softplus() if positive else nn.Identity()
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
        return self.outc(x)
       

class UNet(nn.Module):
    def __init__(self, n_channels, n_features, bilinear=False, positive=False):
        super().__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, n_features))
        self.down1 = (Down(n_features, n_features))
        self.down2 = (Down(n_features, n_features))
        self.down3 = (Down(n_features, n_features))
        factor = 2 if bilinear else 1
        self.down4 = (Down(n_features, n_features // factor))
        self.up1 = (Up(n_features, n_features // factor, bilinear))
        self.up2 = (Up(n_features, n_features // factor, bilinear))
        self.up3 = (Up(n_features, n_features // factor, bilinear))
        self.up4 = (Up(n_features, n_features, bilinear))
        self.outc = nn.Sequential(
            DepthwiseConvolution(n_features, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            DepthwiseConvolution(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            DepthwiseConvolution(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            DepthwiseConvolution(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus() if positive else nn.Identity()
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
        return self.outc(x)
       

class DPTDepthModel(nn.Module):
    def __init__(self, n_features, dino, target_size):
        super().__init__()
        self.dino = dino
        self.head = DPTHead(dino.feature_size, n_features)
        self.out = nn.Upsample(target_size)

    
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, 224, mode="bilinear")
        with torch.no_grad():
            f = self.dino.get_intermediate_features_for_tensor(x, 4)
            f = torch.stack(f).transpose(0, 1)
        return self.out(self.head(f))

    def parameters(self):
        return [*self.head.parameters(), *self.out.parameters()]

    def train(self):
        self.head.train()
        self.out.train()

class ElasticDinoDepthModel(nn.Module):
    def __init__(self, elasticdino):
        super().__init__()
        elasticdino.requires_grad_ = False
        elasticdino.eval()
        self.elasticdino = elasticdino
        self.head = UNet(elasticdino.config["n_features_in"])

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, 224, mode="bilinear")
        with torch.no_grad():
            f = self.elasticdino(x)
        return self.head(f)

    def parameters(self):
        return self.head.parameters()

    def train(self):
        self.head.train()

 

class ElasticDinoDepthModel2(nn.Module):
    def __init__(self, elasticdino, n_features, ed_layer_config):
        super().__init__()
        self.ed_layer_config = ed_layer_config
        self.projections = nn.ModuleList([
            ProjectionLayer(elasticdino.config["n_features_in"], n) for n in ed_layer_config
        ])
        elasticdino.requires_grad_ = False
        elasticdino.eval()
        self.elasticdino = elasticdino
        self.head = UNet(n_features, n_features)

    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, 224, mode="bilinear")
        with torch.no_grad():
            f = self.elasticdino(x, n_hidden_layers=len(self.ed_layer_config))["hidden_layers"]
        f = torch.cat([p(x) for p, x in zip(self.projections, f)], dim=1)
        return self.head(f)

    def parameters(self):
        return self.head.parameters()

    def train(self):
        self.head.train()
    
class UNet2(nn.Module):
    def __init__(self, n_features_in):
        super().__init__()
        self.conv1a=nn.Sequential(nn.Conv2d(n_features_in,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        self.conv1b=nn.Sequential(nn.Conv2d(64,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2a=nn.Sequential(nn.Conv2d(64,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        self.conv2b=nn.Sequential(nn.Conv2d(128,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3a=nn.Sequential(nn.Conv2d(128,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        self.conv3b=nn.Sequential(nn.Conv2d(256,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4a=nn.Sequential(nn.Conv2d(256,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        self.conv4b=nn.Sequential(nn.Conv2d(512,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5a=nn.Sequential(nn.Conv2d(512,1024,3,1,0),nn.BatchNorm2d(1024),nn.ReLU())
        self.conv5b=nn.Sequential(nn.Conv2d(1024,1024,3,1,0),nn.BatchNorm2d(1024),nn.ReLU())
        
        self.up1=PixelShuffle_ICNR(ni=1024, nf=512, scale=2,leaky=False,blur=False)
        self.conv6a=nn.Sequential(nn.Conv2d(1024,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        self.conv6b=nn.Sequential(nn.Conv2d(512,512,3,1,0),nn.BatchNorm2d(512),nn.ReLU())
        
        self.up2=PixelShuffle_ICNR(ni=512, nf=256, scale=2,leaky=False,blur=False)
        self.conv7a=nn.Sequential(nn.Conv2d(512,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        self.conv7b=nn.Sequential(nn.Conv2d(256,256,3,1,0),nn.BatchNorm2d(256),nn.ReLU())
        
        self.up3=PixelShuffle_ICNR(ni=256, nf=128, scale=2,leaky=False,blur=False)
        self.conv8a=nn.Sequential(nn.Conv2d(256,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        self.conv8b=nn.Sequential(nn.Conv2d(128,128,3,1,0),nn.BatchNorm2d(128),nn.ReLU())
        
        self.up4=PixelShuffle_ICNR(ni=128, nf=64, scale=2,leaky=False,blur=False)
        self.conv9a=nn.Sequential(nn.Conv2d(128,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        self.conv9b=nn.Sequential(nn.Conv2d(64,64,3,1,0),nn.BatchNorm2d(64),nn.ReLU())
        
        self.conv9_final=nn.Sequential(nn.Conv2d(64,1,1), nn.Softplus())
        
    def forward(self,input_image):
        conv1=self.conv1b(self.refpad(self.conv1a(self.refpad(input_image))))
        o1=self.maxpool1(conv1)
        
        conv2=self.conv2b(self.refpad(self.conv2a(self.refpad(o1))))
        o2=self.maxpool2(conv2)
        
        conv3=self.conv3b(self.refpad(self.conv3a(self.refpad(o2))))
        o3=self.maxpool3(conv3)
        
        conv4=self.conv4b(self.refpad(self.conv4a(self.refpad(o3))))
        o4=self.maxpool4(conv4)
        
        conv5=self.up1(self.conv5b(self.refpad(self.conv5a(self.refpad(o4)))))
        
        conv6=self.up2(self.conv6b(self.refpad(self.conv6a(self.refpad(self.merge(conv5,conv4))))))
        conv7=self.up3(self.conv7b(self.refpad(self.conv7a(self.refpad(self.merge(conv6,conv3))))))
        conv8=self.up4(self.conv8b(self.refpad(self.conv8a(self.refpad(self.merge(conv7,conv2))))))
        conv9_temp=self.conv9b(self.refpad(self.conv9a(self.refpad(self.merge(conv8,conv1)))))
        conv9=self.conv9_final(conv9_temp)
        
        return conv9
    def refpad(self, x):
        return F.pad(x,(1,1,1,1),'reflect')   
    def merge(self,outputs,inputs):
        offset = outputs.size()[2] - inputs.size()[2]
        
        
        if offset%2!=0:
            padding=2*[offset//2,(offset//2)+1]
        else:
            padding = 2 * [offset // 2, offset // 2]    
        
        output = F.pad(inputs, padding)
        
        return torch.cat([output,outputs],1)