"""
ResNet adapted from:
    https://github.com/kefth/fashion-mnist
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10,
                 size_for_cifar=True, external_fold=1,
                 internal_multiplier=None, do_fold=True):
        """
            Args
            ----
            `external_fold` (int): Number of inputs that will be folded
                                   into a single input to this model. The
                                   corresponding model will have
                                   `external_fold`-times as many input channels
                                   as the original model and will produce
                                   outputs that are `external_fold`-times as
                                   large.

            `internal_multiplier` (int): Multipler on the number of intermediate
                                         channels used in the network. If set to
                                         None, will be set as the square root of
                                         `external_fold`.
        """
        self.fold = external_fold
        if internal_multiplier is not None:
            self.internal_multiplier = internal_multiplier
        else:
            sqrt_fold = math.sqrt(self.fold)
            if not sqrt_fold.is_integer():
                raise Exception("Specified fold {} is not a square".format(external_fold))
            self.internal_multiplier = sqrt_fold

        self.inplanes = int(64 * self.internal_multiplier)
        super(ResNet, self).__init__()
        self.size_for_cifar = size_for_cifar
        if size_for_cifar:
            num_channels = 3
        else:
            num_channels = 1

        if do_fold:
            num_channels *= self.fold
            num_classes *= self.fold

        self.conv1 = nn.Conv2d(num_channels, int(64 * self.internal_multiplier),
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * self.internal_multiplier))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, int(64 * self.internal_multiplier), layers[0])
        self.layer2 = self._make_layer(block, int(128 * self.internal_multiplier), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * self.internal_multiplier), layers[2], stride=2)

        if size_for_cifar:
            self.layer4 = self._make_layer(block, int(512 * self.internal_multiplier), layers[3], stride=2)
            if do_fold:
                self.avgpool = nn.AvgPool2d(4)
            else:
                self.avgpool = nn.AvgPool2d(4, padding=1)
            self.fc = nn.Linear(int(512 * block.expansion * self.internal_multiplier), num_classes)
        else:
            if do_fold:
                self.avgpool = nn.AvgPool2d(7)
            else:
                self.avgpool = nn.AvgPool2d(7, padding=1)
            self.fc = nn.Linear(int(256 * block.expansion * self.internal_multiplier), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # We only use the final layer if `x` started as (-1, 3, 32, 32)
        if self.size_for_cifar:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet18(ResNet):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10,
                 size_for_cifar=True, external_fold=1, internal_multiplier=None,
                 do_fold=True):
        super(ResNet18, self).__init__(block=block, layers=layers,
                                     num_classes=num_classes,
                                     size_for_cifar=size_for_cifar,
                                     external_fold=external_fold,
                                     internal_multiplier=internal_multiplier,
                                     do_fold=do_fold)
