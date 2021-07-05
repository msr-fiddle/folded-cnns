"""
CNN architectures used for transforming the game-scraping CNNs
using the compond-scaling procedures proposed in EfficientNets [1].

[1] https://arxiv.org/abs/1905.11946
"""

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvN(nn.Module):
    def __init__(self, cfgs, fold, depth_mult=1.2, width_mult=1.1, res_mult=1.15, do_fold=True):
        """
        NOTE: We overload `fold` with the parameter \phi in the EfficientNet paper [1], and
        `do_fold` with whether or not EfficientNet compound scaling should be performed.

        [1] https://arxiv.org/abs/1905.11946
        """
        super().__init__()

        model_arch_template = cfgs['model-architecture']['template']
        model_arch = cfgs['model-architecture']['params']
        channel_mult = (1. / width_mult) ** (fold - 1) if do_fold else 1
        num_inner = 3 if model_arch_template == '1_3s_1_convnet' else 1
        nconvs = 1 + (model_arch['n-conv-units'] * num_inner) + 1
        depth_scale = (1. / width_mult) ** (fold - 1) if do_fold else 1
        nconvs = int(nconvs * depth_scale) if do_fold else nconvs
        convs_so_far = 0

        nclasses = cfgs['num-classes']
        in_channels = cfgs['chan-in']
        layers = []

        int_channels = int(model_arch['conv1_n_feature_maps'] * channel_mult)

        if convs_so_far < nconvs:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=int_channels,
                                    kernel_size=3,
                                    padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            convs_so_far += 1

        for i in range(model_arch['n-conv-units']):
            conv_added = False
            if convs_so_far < nconvs:
                next_int_channels = int(model_arch['convi_n_feature_maps'] * channel_mult)
                layers.append(nn.Conv2d(in_channels=int_channels,
                                        out_channels=next_int_channels,
                                        kernel_size=3,
                                        padding=1))
                int_channels = next_int_channels
                layers.append(nn.ReLU())
                convs_so_far += 1
                conv_added = True

            if model_arch_template == '1_3s_1_convnet':
                if convs_so_far < nconvs:
                    layers.append(nn.Conv2d(in_channels=int_channels,
                                            out_channels=int_channels,
                                            kernel_size=3,
                                            padding=1))
                    layers.append(nn.ReLU())
                    convs_so_far += 1
                    conv_added = True
                if convs_so_far < nconvs:
                    layers.append(nn.Conv2d(in_channels=int_channels,
                                            out_channels=int_channels,
                                            kernel_size=3,
                                            padding=1))
                    layers.append(nn.ReLU())
                    convs_so_far += 1
                    conv_added = True

            if conv_added:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1))

        if convs_so_far < nconvs:
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
        res_scale = (1. / res_mult) ** (fold - 1) if do_fold else 1
        height = int(cfgs['out-h'] * res_scale)
        width = int(cfgs['out-w'] * res_scale)
        dummy_input = torch.rand(1, in_channels, height, width)
        dummy_output = nn.Sequential(*layers)(dummy_input)

        self.lin = nn.Linear(dummy_output.size(-1), nclasses)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        conv = self.conv(x)
        out = self.lin(conv)
        return out
