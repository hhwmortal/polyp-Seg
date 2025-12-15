import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50
import torch.nn.functional as F
import math
from mobilenetv4 import MobileNetV4ConvSmall


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


# 融合的RCA
class Conv2D_RCA_Dynamic(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1, dilation=1, bias=True, act=True,
                 ratio=1, band_kernel_size=11, square_kernel_size=3):
        super().__init__()

        self.conv = Conv2D(in_c, out_c, kernel_size, padding, stride, dilation, bias, act)
        self.rca = RCA(out_c, ratio=ratio, band_kernel_size=band_kernel_size, square_kernel_size=square_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        att = self.rca(x)
        x = x * att  # 让 RCA 作为动态权重调制 Conv2D 输出
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        s = self.shortcut(inputs)
        return self.relu(x + s)


class residual_transformer_block(nn.Module):
    def __init__(self, in_c, out_c, patch_size=4, num_heads=4, num_layers=2, dim=None):
        super().__init__()

        self.ps = patch_size
        self.c1 = Conv2D(in_c, out_c)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.te = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.c2 = Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.c3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.r1 = residual_block(out_c, out_c)

    def forward(self, inputs):
        x = self.c1(inputs)

        b, c, h, w = x.shape
        num_patches = (h * w) // (self.ps ** 2)
        x = torch.reshape(x, (b, (self.ps ** 2) * c, num_patches))
        x = self.te(x)
        x = torch.reshape(x, (b, c, h, w))

        x = self.c2(x)
        s = self.c3(inputs)
        x = self.relu(x + s)
        x = self.r1(x)
        return x


# 定义h_sigmoid激活函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


# 定义h_swish激活函数
class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# 定义Coordinate Attention模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 水平和垂直方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向

        mip = max(8, inp // reduction)  # 计算中间层的通道数

        # 1x1卷积降维
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 1x1卷积用于计算注意力权重
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 动态调整注意力强度
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接和特征压缩
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分割回原始方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 动态调整注意力权重
        out = identity * (a_w * a_h) + identity * (1 - a_w * a_h) * self.alpha

        return out


class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(MultiScaleSpatialAttention, self).__init__()
        hidden_channels = max(4, in_channels // rate)
        self.conv3 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(hidden_channels * 3)
        self.relu = nn.ReLU()
        self.conv_fuse = nn.Conv2d(hidden_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        feat7 = self.conv7(x)
        feat = torch.cat([feat3, feat5, feat7], dim=1)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.conv_fuse(feat)
        return torch.sigmoid(feat)


class ACSSA(nn.Module):
    def __init__(self, in_channels, rate=4, oup=None):
        super(ACSSA, self).__init__()
        # 如果没有传入 oup 参数，使用 in_channels 作为默认值
        oup = oup or in_channels  # 默认值为 in_channels
        self.CoordAtt = CoordAtt(in_channels, oup)  # 这里传入 oup 参数
        self.spatial_attention = MultiScaleSpatialAttention(in_channels, rate)

    def channel_shuffle(self, x, groups):
        B, C, H, W = x.size()
        channels_per_group = C // groups
        x = x.view(B, groups, channels_per_group, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(B, -1, H, W)

    def forward(self, x):
        # 通道注意力模块
        ch_att = self.CoordAtt(x)
        x = x * ch_att

        # 自适应分组（示例中使用固定分组，可以考虑设计动态分组）
        groups = max(2, x.size(1) // 16)
        x = self.channel_shuffle(x, groups)

        # 空间注意力模块
        sp_att = self.spatial_attention(x)
        out = x * sp_att
        return out


class RCA(nn.Module):

    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=3, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        backbone = MobileNetV4ConvSmall()
        self.layer0 = backbone.conv0
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.e1 = Conv2D_RCA_Dynamic(64, 64, kernel_size=1, padding=0)
        self.e2 = Conv2D_RCA_Dynamic(128, 64, kernel_size=1, padding=0)
        self.e3 = Conv2D_RCA_Dynamic(256, 64, kernel_size=1, padding=0)
        self.e4 = Conv2D_RCA_Dynamic(512, 64, kernel_size=1, padding=0)

        self.ACSSA = ACSSA(128)

        """ Decoder """
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = residual_transformer_block(64 + 64, 64, dim=256)
        self.r2 = residual_transformer_block(64 + 64, 64, dim=1024)
        self.r3 = residual_block(64 + 64, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        x0 = inputs  # [-1, 3, 512, 512]
        x1 = self.layer0(x0)  # [-1, 64, h/2, w/2]

        x2 = self.layer1(x1)  # [-1, 128, h/4, w/4]
        x2 = self.ACSSA(x2)

        x3 = self.layer2(x2)  # [-1, 256, h/8, w/8]

        x4 = self.layer3(x3)  # [-1, 512, h/16, w/16]

        e1 = self.e1(x1)
        e2 = self.e2(x2)
        e3 = self.e3(x3)
        e4 = self.e4(x4)

        """ Decoder """
        x = self.up(e4)
        x = torch.cat([x, e3], axis=1)  # [-1, 128, 64, 64]
        x = self.r1(x)  # [-1, 64, 64, 64]

        x = self.up(x)  # [-1, 64, 128, 128]
        x = torch.cat([x, e2], axis=1)  # [-1, 128, 128, 128]
        x = self.r2(x)  # [-1, 64, 128, 128]

        x = self.up(x)  # [-1, 64, 256, 256]
        x = torch.cat([x, e1], axis=1)  # [-1, 128, 256, 256]
        x = self.r3(x)  # [-1, 64, 256, 256]

        x = self.up(x)  # [-1, 64, 512, 512]

        """ Classifier """
        outputs = self.outputs(x)
        return outputs


if __name__ == "__main__":
    x = torch.randn((4, 3, 512, 512))
    model = Model()
    y = model(x)
    print(y.shape)
