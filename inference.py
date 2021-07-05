import argparse
import json
import os
import time
import torch
try:
    from torch2trt import torch2trt
except:
    print("Could not import torch2trt")

import models

game_models = ["game_" + x for x in [
          "lol/goldnumber-fraction",
          "apex/squad_count_v2",
          "sot/coin_count-digits_3_4-v2",
          "sot/timer3",
          "lol/goldnumber-int",
          "lol/timer-minutes",]]

efficientnet_game_models = ["efficientnet-game_" + x for x in [
          "lol/goldnumber-fraction",
          "apex/squad_count_v2",
          "sot/coin_count-digits_3_4-v2",
          "sot/timer3",
          "lol/goldnumber-int",
          "lol/timer-minutes",]]

noscope_models = [
        "noscope_coral",
        "noscope_night",
        "noscope_taipei",
        "noscope_roundabout"
        ]


all_models = game_models + noscope_models + efficientnet_game_models


def get_model(args):
    if "efficientnet" in args.model and "game" in args.model:
        # Model name must now be of the form "game_<game_name>/<game_name>"
        # Example: "efficientnet-game_lol/goldnumber-int"
        subdir = '_'.join(args.model.split('_')[1:])
        config_dir = "config/" + subdir

        desc_file = os.path.join(config_dir, 'model_description.json')
        with open(desc_file) as infile:
            mdesc = json.load(infile)

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
                                    args.fold,
                                    depth_mult=args.depth_mult,
                                    width_mult=args.width_mult,
                                    res_mult=args.res_mult,
                                    do_fold=args.do_fold)

        in_channels = 3
        in_size = (in_channels, int(out_h / args.res_mult), int(out_w / args.res_mult))

    elif "game" in args.model:
        # Model name must now be of the form "game_<game_name>/<game_name>"
        # Example: "game_lol/goldnumber-int"
        subdir = '_'.join(args.model.split('_')[1:])
        config_dir = "config/" + subdir

        desc_file = os.path.join(config_dir, 'model_description.json')
        with open(desc_file) as infile:
            mdesc = json.load(infile)

        out_h = mdesc['out-h']
        out_w = mdesc['out-w']
        cfgs = {
            'out-h': out_h,
            'out-w': out_w,
            'chan-in': 3,
            'num-classes': len(mdesc["ClassNames"]),
            'model-architecture': mdesc['model-architecture']
        }

        channels_multiplier = (args.fold ** 0.5)
        model = models.game.ConvN(cfgs, fold=args.fold,
                                    channel_mult=channels_multiplier,
                                    do_fold=args.do_fold)

        in_channels = 3
        if args.do_fold:
            in_channels *= args.fold
        in_size = (in_channels, out_h, out_w)

    elif "noscope" in args.model:
        # Model name must now be of the form "noscope_<name>"
        # Example: "noscope_taipei"
        model_name = args.model.split('noscope_')[-1]
        channels_multiplier = (args.fold ** 0.5)
        model = models.noscope.noscope_cnn(model_name,
                                           fold=args.fold,
                                           internal_multiplier=channels_multiplier,
                                           do_fold=args.do_fold)

        in_h = 50
        in_w = 50
        in_channels = 3
        if args.do_fold:
            in_channels *= args.fold
        in_size = (in_channels, in_h, in_w)
    else:
        raise Exception("Unrecognized model '{}'".format(args.model))

    return model, in_size


def benchmark(args):
    s = time.time()
    model, in_size = get_model(args)
    model = model.eval().cuda()
    e = time.time()
    in_size = (args.batch_size, *in_size)
    sample = torch.rand(in_size).cuda()

    if args.trt == 1:
        use_fp16 = (args.fp16 == 1)
        model = torch2trt(model, [sample], max_batch_size=args.batch_size,
                          fp16_mode=use_fp16)

    with torch.no_grad():
        for i in range(args.num_warmup):
            out = model(sample)
        torch.cuda.synchronize()

        start = time.time()
        for i in range(args.num_iteration):
            out = model(sample)
        torch.cuda.synchronize()
        end = time.time()

    num_images = args.num_iteration * args.batch_size
    if args.do_fold and "efficientnet" not in args.model:
        num_images *= args.fold
    total_time = end - start
    print("{:.2f}".format(num_images / total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str,
                        help="Model to perform inference over")
    parser.add_argument("fold", type=int)
    parser.add_argument("--do_fold", action="store_true",
                        help="Whether to perform folding")
    parser.add_argument("--fold", type=int, default=4,
                        help="Number of images to fold into a single input")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size to use")
    parser.add_argument("--num_iteration", type=int, default=10000,
                        help="Number of iterations to run")
    parser.add_argument("--num_warmup", type=int, default=100,
                        help="Number of warmup iterations to run")
    parser.add_argument("--depth_mult", type=float, default=1.)
    parser.add_argument("--width_mult", type=float, default=1.)
    parser.add_argument("--res_mult", type=float, default=1.)
    parser.add_argument("--trt", type=int, default=1)
    parser.add_argument("--fp16", type=int, default=1, choices=[0,1])
    args = parser.parse_args()

    benchmark(args)
