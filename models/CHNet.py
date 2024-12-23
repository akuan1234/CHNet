import torch
import torch.nn as nn
import torch.nn.functional as F
from smt import smt_t

from thop import profile
from torch import Tensor
from typing import List
from einops import rearrange

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class NFI(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self,
                 op_channel: int = 64,
                 alpha: float = 1/2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = 32
        self.low_channel = low_channel = 32
        self.squeeze1 = nn.Conv2d(up_channel, up_channel//squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel//squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel//squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=1, groups = 1)
        self.PWC1 = nn.Conv2d(up_channel//squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.fuse = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        y1 = self.GWC(up) + self.PWC1(up)
        y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([y1, y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        out = out1 + out2
        out = self.fuse(out)
        out = out + x
        return self.final_relu(out)

class FG(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(FG, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)
        self.fuse = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.final_relu = nn.ReLU(True)

    def forward(self, y):
        b, c, h, w = y.shape
        x = y.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)
        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)
        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
        output = self.fuse(output)
        out = output + y
        return self.final_relu(out)

class FE(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FE, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv2d(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv2d(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv2d((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv2d(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv2d((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv2d(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv2d(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        out = out + x
        return self.relu(out)

class CHNet(nn.Module):
    def __init__(self,
                op_channel:int=64,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super(CHNet, self).__init__()

        self.smt = smt_t()

        self.Translayer2_1 = BasicConv2d(128, 64, 1)
        self.Translayer3_1 = BasicConv2d(256, 64, 1)
        self.Translayer4_1 = BasicConv2d(512, 64, 1)

        self.NFI = NFI(64,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

        self.FE = FE(in_planes=64, out_planes=64)
        self.FG = FG(dim=64, num_heads=8)

        self.uconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.uconv3 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1, relu=True)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # print('01', datetime.now())
        rgb_list = self.smt(x)

        r1 = rgb_list[3]  # 512,12
        r2 = rgb_list[2]  # 256,24
        r3 = rgb_list[1]  # 128,48
        r4 = rgb_list[0]  # 64,96,96

        r3 = self.Translayer2_1(r3)  #
        r2 = self.Translayer3_1(r2)
        r1 = self.Translayer4_1(r1)  # 
        r2 = self.FE(r2)
        r3 = self.FE(r3)
        r4 = self.FE(r4)

        r1 = self.up1(r1)
        x12 = self.NFI(r2 + r1)
        r2 = self.up1(r2)
        x23 = self.NFI(r3 + r2)
        r3 = self.up1(r3)
        x34 = self.NFI(r4 + r3)

        x1 = self.FG(r1 + x12)
        x1 = self.up1(x1)
        x2 = self.FG(x1 + x23)
        x2 = self.up1(x2)
        x3 = self.FG(x2 + x34)

        r123 = F.interpolate(self.predtrans1(x3), size=416, mode='bilinear')
        r12 = F.interpolate(self.predtrans2(x2), size=416, mode='bilinear')
        r1 = F.interpolate(self.predtrans3(x1), size=416, mode='bilinear')

        return r123, r12, r1

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")


if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    flops, params = profile(CHNet(x), (x,))
    print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
