""" CNN architectures used for the NoScope tasks """

import torch
import torch.nn as nn

from models.game import Flatten


# These parameters are taken from Table 2 of the NoScope VLDB paper:
#   http://www.vldb.org/pvldb/vol10/p1586-kang.pdf
model_params = {
        "coral": {
            "nb_classes": 2,
            "nb_dense": 128,
            "nb_filters": 16,
            "nb_layers": 2
        },
        "night": {
            "nb_classes": 2,
            "nb_dense": 128,
            "nb_filters": 16,
            "nb_layers": 2
        },
        "taipei": {
            "nb_classes": 2,
            "nb_dense": 32,
            "nb_filters": 64,
            "nb_layers": 2
        },
        "roundabout": {
            "nb_classes": 2,
            "nb_dense": 32,
            "nb_filters": 32,
            "nb_layers": 4
        }
}


def noscope_layers(nb_classes, nb_dense, nb_filters, nb_layers,
        kernel_size=3, stride=1,
        fold=1., internal_multiplier=None, do_fold=False,
        true_layers=False, do_dropout=False):

    layers = []

    if internal_multiplier is None:
        internal_multiplier = (fold ** 0.5)

    in_channels = 3

    if do_fold:
        in_channels *= int(fold)
        nb_classes *= int(fold)

    # Based on https://github.com/stanford-futuredata/noscope/blob/ffc53d415a6075258a766c01621abcc65ff71200/noscope/Models.py
    first_int_channels = nb_filters
    middle_int_channels = 2 * nb_filters

    # Apply multiplier
    first_int_channels = int(first_int_channels * internal_multiplier)
    middle_int_channels = int(middle_int_channels * internal_multiplier)
    num_dense = int(nb_dense * internal_multiplier)

    layers.append(nn.Conv2d(in_channels=in_channels,
                            out_channels=first_int_channels,
                            kernel_size=kernel_size,
                            padding=1))
    layers.append(nn.ReLU())

    layers.append(nn.Conv2d(in_channels=first_int_channels,
                            out_channels=first_int_channels,
                            kernel_size=kernel_size,
                            padding=1))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if do_dropout:
        layers.append(nn.Dropout(0.25))

    if nb_layers > 2:
        layers.append(nn.Conv2d(in_channels=first_int_channels,
                                out_channels=middle_int_channels,
                                kernel_size=kernel_size,
                                padding=1))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=middle_int_channels,
                                out_channels=middle_int_channels,
                                kernel_size=kernel_size,
                                padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if do_dropout:
            layers.append(nn.Dropout(0.25))
        dense_in = middle_int_channels
    else:
        dense_in = first_int_channels

    # Hack to figure out the number of input features for the fully-connected
    # layer
    in_h = 50
    in_w = 50
    dummy_input = torch.rand(1, in_channels, in_h, in_w)
    dummy_output = nn.Sequential(*layers)(dummy_input)
    cur_h, cur_w =  dummy_output.size(2), dummy_output.size(3)

    layers.append(Flatten())

    layers.append(nn.Linear(dense_in * cur_h * cur_w, num_dense))
    layers.append(nn.ReLU())

    if do_dropout:
        layers.append(nn.Dropout(0.25))

    layers.append(nn.Linear(num_dense, nb_classes))

    return layers


def noscope_cnn(model_name, do_fold, fold, internal_multiplier=None,
                do_dropout=False):
    if internal_multiplier is None:
        internal_multiplier = (fold ** 0.5)
    layers = noscope_layers(**model_params[model_name], do_fold=do_fold,
            fold=fold, internal_multiplier=internal_multiplier,
            true_layers=True, do_dropout=do_dropout)

    return nn.Sequential(*layers)
