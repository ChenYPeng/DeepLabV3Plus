import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
           'dilated_densenet121', 'dilated_densenet161', 'dilated_densenet169', 'dilated_densenet201']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1, batch_norm=None):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', batch_norm(num_input_features)),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, 1, 1, bias=False)),
        self.add_module('norm2', batch_norm(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, dilation, dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size,
                 growth_rate, drop_rate, dilation=1, batch_norm=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, dilation, batch_norm=batch_norm)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, batch_norm=None):
        super(_Transition, self).__init__()
        self.add_module('norm', batch_norm(num_input_features))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 1, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, 2))


# Net
class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), out_featureres=(3, 5), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, batch_norm=None, **kwargs):
        super(DenseNet, self).__init__()
        self.out_featureres = out_featureres
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, 7, 2, 2, bias=False)),
            ('norm0', batch_norm(num_init_features)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(3, 2, 1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, batch_norm=batch_norm)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2, batch_norm=batch_norm)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.num_features = num_features

        # Final batch norm
        self.features.add_module('norm5', batch_norm(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        SB = self.features[:3](x)  # conv0, norm0,  relu0
        DB1 = self.features[3:5](SB)  # pool0, denseblock1
        DB2 = self.features[5:7](DB1)  # transition1, denseblock2
        DB3 = self.features[7:9](DB2)  # transition2, denseblock3
        DB4 = self.features[9::](DB3)  # transition3,denseblock4, norm5
        # out = F.relu(DB4, True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(DB4.size(0), -1)
        # out = self.classifier(out)
        return DB2, DB4


class DilatedDenseNet(DenseNet):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, output_stride=8, batch_norm=None, **kwargs):
        super(DilatedDenseNet, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate,
                                              num_classes, batch_norm)
        assert (output_stride == 8 or output_stride == 16), "dilate_scale can only set as 8 or 16"
        from functools import partial
        if output_stride == 8:
            self.features.denseblock3.apply(partial(self._conv_dilate, dilate=2))
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=4))
            del self.features.transition2.pool
            del self.features.transition3.pool
        elif output_stride == 16:
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=2))
            del self.features.transition3.pool

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.padding = (dilate, dilate)
                m.dilation = (dilate, dilate)


# Specification
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (96, 48, [6, 12, 48, 32]),
                 264: (64, 32, [6, 12, 64, 48])}


# Constructor
def get_densenet(num_layers, pretrained=False, batch_norm=None, **kwargs):
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DenseNet(growth_rate, block_config, num_init_features, batch_norm=batch_norm, **kwargs)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def get_dilated_densenet(num_layers, output_stride, pretrained=False, batch_norm=None, **kwargs):
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    model = DilatedDenseNet(growth_rate, block_config, num_init_features, output_stride=output_stride,
                            batch_norm=batch_norm)
    if pretrained:
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet121(num_layers=121, pretrained=True, batch_norm=SynchronizedBatchNorm2d, **kwargs):
    return get_densenet(num_layers, pretrained, batch_norm, **kwargs)


def densenet161(num_layers=161, pretrained=True, batch_norm=SynchronizedBatchNorm2d, **kwargs):
    return get_densenet(num_layers, pretrained, batch_norm, **kwargs)


def densenet169(num_layers=169, pretrained=True, batch_norm=SynchronizedBatchNorm2d, **kwargs):
    return get_densenet(num_layers, pretrained, batch_norm, **kwargs)


def densenet201(num_layers=201, pretrained=True, batch_norm=SynchronizedBatchNorm2d, **kwargs):
    return get_densenet(num_layers, pretrained, batch_norm, **kwargs)


def dilated_densenet121(num_layers=121, output_stride=16, pretrained=True, batch_norm=SynchronizedBatchNorm2d,
                        **kwargs):
    return get_dilated_densenet(num_layers, output_stride, pretrained, batch_norm, **kwargs)


def dilated_densenet161(num_layers=161, output_stride=16, pretrained=True, batch_norm=SynchronizedBatchNorm2d,
                        **kwargs):
    return get_dilated_densenet(num_layers, output_stride, pretrained, batch_norm, **kwargs)


def dilated_densenet169(num_layers=169, output_stride=16, pretrained=True, batch_norm=SynchronizedBatchNorm2d,
                        **kwargs):
    return get_dilated_densenet(num_layers, output_stride, pretrained, batch_norm, **kwargs)


def dilated_densenet201(num_layers=201, output_stride=16, pretrained=True, batch_norm=SynchronizedBatchNorm2d,
                        **kwargs):
    return get_dilated_densenet(num_layers, output_stride, pretrained=pretrained, batch_norm=batch_norm, **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 513, 513)
    model = densenet121()
    out = model(img)
    for o in out:
        print(o.shape)
