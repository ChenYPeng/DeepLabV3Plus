import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


# ---------------------------------------------------#
#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
# ---------------------------------------------------#

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, BatchNorm=None):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size-1) // 2, bias=False)
        self.bn = BatchNorm(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None, BatchNorm=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, kernel_size=1, BatchNorm=BatchNorm),
            BasicConv(hidden_channels, channels, kernel_size=3, BatchNorm=BatchNorm)
        )

    def forward(self, x):
        return x + self.block(x)


# --------------------------------------------------------------------#
#   CSPdarknet的结构块
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
# --------------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first, BatchNorm=None):
        super(Resblock_body, self).__init__()
        # ----------------------------------------------------------------#
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        # ----------------------------------------------------------------#
        self.downsample_conv = BasicConv(in_channels, out_channels, kernel_size=3, stride=2, BatchNorm=BatchNorm)

        if first:
            # --------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            # --------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels, kernel_size=1, BatchNorm=BatchNorm)

            # ----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            # ----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels, kernel_size=1, BatchNorm=BatchNorm)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels // 2, BatchNorm=BatchNorm),
                BasicConv(out_channels, out_channels, kernel_size=1, BatchNorm=BatchNorm)
            )

            self.concat_conv = BasicConv(out_channels * 2, out_channels, kernel_size=1, BatchNorm=BatchNorm)
        else:
            # --------------------------------------------------------------------------#
            #   然后建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            # --------------------------------------------------------------------------#
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, kernel_size=1, BatchNorm=BatchNorm)

            # ----------------------------------------------------------------#
            #   主干部分会对num_blocks进行循环，循环内部是残差结构。
            # ----------------------------------------------------------------#
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, kernel_size=1, BatchNorm=BatchNorm)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels // 2, BatchNorm=BatchNorm) for _ in range(num_blocks)],
                BasicConv(out_channels // 2, out_channels // 2, kernel_size=1, BatchNorm=BatchNorm)
            )

            self.concat_conv = BasicConv(out_channels, out_channels, kernel_size=1, BatchNorm=BatchNorm)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        # ------------------------------------#
        #   将大残差边再堆叠回来
        # ------------------------------------#
        x = torch.cat([x1, x0], dim=1)
        # ------------------------------------#
        #   最后对通道数进行整合
        # ------------------------------------#
        x = self.concat_conv(x)

        return x


# ---------------------------------------------------#
#   CSPdarknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
# ---------------------------------------------------#
class CSPDarkNet(nn.Module):
    def __init__(self, layers, out_indices, pretrained=True, BatchNorm=None):
        super(CSPDarkNet, self).__init__()
        self.out_features = out_indices
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1, BatchNorm=BatchNorm)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True, BatchNorm=BatchNorm),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False,
                          BatchNorm=BatchNorm),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False,
                          BatchNorm=BatchNorm),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False,
                          BatchNorm=BatchNorm),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False,
                          BatchNorm=BatchNorm)
        ])

        self.num_features = 1

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = torch.load('C:/Users/85782/.cache/torch/hub/checkpoints/yolo4_voc_weights.pth')

        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            k = k[9:]
            if k in state_dict:
                model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        for idx, layer in enumerate(self.stages):
            x = layer(x)
            if idx + 1 in self.out_features:
                outputs.append(x)
        return outputs


def darknet53(layers=(1, 2, 8, 8, 4), out_indices=(3, 5), pretrained=True, BatchNorm=None, **kwargs):
    model = CSPDarkNet(layers, out_indices, pretrained, BatchNorm=BatchNorm)
    return model


if __name__ == '__main__':
    net = darknet53(layers=[1, 2, 8, 8, 4], out_indices=(1, 2, 3, 4, 5), pretrained=True, BatchNorm=SynchronizedBatchNorm2d)
    img = torch.rand(1, 3, 512, 512)
    out = net(img)
    for x in out:
        print(x.shape)
