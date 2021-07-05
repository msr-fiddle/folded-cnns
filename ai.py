"""
Prints the arithmetic intensities of FoldedCNNs and EfficientNet-transformed
specialized CNNs.
"""

import json
import os
import torch
import torchvision

import models


def conv_hook(mod, indata, outdata):
    assert len(indata) == 1
    indata = indata[0]
    batch_size = outdata.size(0)
    c_in = mod.in_channels
    c_out = mod.out_channels
    groups = mod.groups
    k_h, k_w = mod.kernel_size
    do_bias = mod.bias is not None
    o_h = outdata.size(2)
    o_w = outdata.size(3)

    mod.ops = 2 * (batch_size * k_h * k_w * o_h * o_w * c_in * c_out) / groups
    if do_bias:
        mod.ops += (batch_size * o_h * o_w * c_out)

    mod.in_elts = indata.view(-1).size(0)
    mod.out_elts = outdata.view(-1).size(0)
    mod.model_elts = k_h * k_w * c_in * c_out / groups


def lin_hook(mod, indata, outdata):
    assert len(indata) == 1
    indata = indata[0]
    batch_size = outdata.size(0)
    f_in = mod.in_features
    f_out = mod.out_features
    do_bias = mod.bias is not None

    mod.ops = 2 * batch_size * f_in * f_out
    if do_bias:
        mod.ops += (batch_size * f_out)

    mod.in_elts = indata.view(-1).size(0)
    mod.out_elts = outdata.view(-1).size(0)
    mod.model_elts = f_in * f_out


def add_ai_hooks(model, insize, verbose=False):
    mod_ais = []
    def add_hook(mod):
        if isinstance(mod, torch.nn.Conv2d):
            mod.register_forward_hook(conv_hook)
            mod_ais.append(mod)
        elif isinstance(mod, torch.nn.Linear):
            mod.register_forward_hook(lin_hook)
            mod_ais.append(mod)
        elif verbose:
            print("Not registering for mod", type(mod))

    model.apply(add_hook)

    indata = torch.rand(insize)
    with torch.no_grad():
        model(indata)

    return mod_ais


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes_per_elt", default=2.,
                        help="Number of bytes per element of inputs, outputs, and model")
    args = parser.parse_args()

    game_models = [
        "lol/goldnumber-fraction",
        "apex/squad_count_v2",
        "sot/coin_count-digits_3_4-v2",
        "sot/timer3",
    ]

    noscope_models = [
            "coral",
            "night",
            "taipei",
            "roundabout"
    ]

    print("model,batch_size,original_AI,tformed_AI_2,tformed_AI_3,tformed_AI_4")
    for model_group, group_name in [(game_models, "game"), (noscope_models, "noscope"), (game_models, "efficientnet")]:
        for model_name in model_group:
            overall_name=group_name + "-" + model_name
            for batch_size in [1024]:
                fold_ais = []
                folds = [1, 2, 3, 4]
                for fold in folds:
                    if group_name == "game":
                        cfg_file = os.path.join("config", model_name, "model_description.json")
                        with open(cfg_file, 'r') as infile:
                           model_desc = json.load(infile)

                        cfgs = {
                                'out-h': model_desc['out-h'],
                                'out-w': model_desc['out-w'],
                                'chan-in': 3,
                                'num-classes': len(model_desc["ClassNames"]),
                                'model-architecture': model_desc['model-architecture']
                                }

                        model = models.game.ConvN(cfgs, fold=fold, do_fold=(fold != 1))
                        insize = (1, 3*fold, cfgs["out-h"], cfgs["out-w"])
                    elif group_name == "efficientnet":
                        if fold > 2:
                            continue
                        config_dir = os.path.join("config", model_name)
                        desc_file = os.path.join(config_dir, 'model_description.json')
                        with open(desc_file) as infile:
                            mdesc = json.load(infile)

                        if fold > 1:
                            depth_mult=1.2
                            width_mult=1.1
                            res_mult=1.15
                        else:
                            depth_mult=1
                            width_mult=1
                            res_mult=1

                        out_h = mdesc['out-h']
                        out_w = mdesc['out-w']
                        cfgs = {
                                'out-h': out_h,
                                'out-w': out_w,
                                'chan-in': 3,
                                'num-classes': len(mdesc["ClassNames"]),
                                'model-architecture': mdesc['model-architecture']
                                }

                        model = models.efficientnet_game.ConvN(cfgs,
                                fold,
                                depth_mult=depth_mult,
                                width_mult=width_mult,
                                res_mult=res_mult,
                                do_fold=(fold != 1))

                        in_channels = 3
                        insize = (1, in_channels, int(out_h / res_mult), int(out_w / res_mult))
                    elif group_name == "noscope":
                        model = models.noscope.noscope_cnn(model_name, do_fold=(fold != 1), fold=fold)
                        insize = (1, 3*fold, 50, 50)

                    mod_ais = add_ai_hooks(model, insize)
                    ops = sum([m.ops for m in mod_ais]) * batch_size / fold
                    model_elts = sum([m.model_elts for m in mod_ais])
                    in_out_elts = sum([m.in_elts + m.out_elts for m in mod_ais]) * batch_size / fold
                    elts = model_elts + in_out_elts
                    fold_ais.append(ops / (args.bytes_per_elt * elts))

                print(overall_name + "," + str(batch_size) + "," + ",".join(["{:.2f}".format(f) for f in fold_ais]))
