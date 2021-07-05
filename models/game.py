""" CNN architectures used for the game-scraping tasks """

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvN(nn.Module):
    def __init__(self, cfgs, fold, channel_mult=None, do_fold=True):
        super().__init__()

        sqrt_fold = (fold ** 0.5)
        if channel_mult is not None:
            channel_mult = channel_mult
        else:
            channel_mult = sqrt_fold

        model_arch_template = cfgs['model-architecture']['template']
        model_arch = cfgs['model-architecture']['params']
        nclasses = cfgs['num-classes']
        in_channels = cfgs['chan-in']
        if do_fold:
            in_channels *= fold
            nclasses *= fold

        layers = []

        int_channels = int(model_arch['conv1_n_feature_maps'] * channel_mult)

        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=int_channels,
                                kernel_size=3,
                                padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        for i in range(model_arch['n-conv-units']):
            next_int_channels = int(model_arch['convi_n_feature_maps'] * channel_mult)
            layers.append(nn.Conv2d(in_channels=int_channels,
                                    out_channels=next_int_channels,
                                    kernel_size=3,
                                    padding=1))
            int_channels = next_int_channels
            layers.append(nn.ReLU())

            if model_arch_template == '1_3s_1_convnet':
                layers.append(nn.Conv2d(in_channels=int_channels,
                                        out_channels=int_channels,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(in_channels=int_channels,
                                        out_channels=int_channels,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.ReLU())

            layers.append(nn.MaxPool2d(kernel_size=3, stride=1))

        layers.append(nn.Conv2d(in_channels=int_channels,
                                out_channels=int(model_arch['convn_n_feature_maps'] * channel_mult),
                                kernel_size=3,
                                padding=1))

        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(Flatten())

        # Hack to figure out what the input size of the final linear layer is.
        # Just run all of the previous layers on a dummy input and see what the
        # resulting size is.
        height = cfgs['out-h']
        width = cfgs['out-w']
        dummy_input = torch.rand(1, in_channels, height, width)
        dummy_output = nn.Sequential(*layers)(dummy_input)

        self.lin = nn.Linear(dummy_output.size(-1), nclasses)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        conv = self.conv(x)
        out = self.lin(conv)
        return out
