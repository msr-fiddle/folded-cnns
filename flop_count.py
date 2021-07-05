import argparse
import json
import os
import torch
from thop import profile # pip3 install thop

from models import efficientnet_game, game, noscope

efficientnet_game_models = [ "efficientnet-game_" + x for x in [
        "lol/goldnumber-fraction",
        "lol/goldnumber-int",
        "lol/timer-minutes",
        "apex/squad_count_v2",
        "sot/coin_count-digits_3_4-v2",
        "sot/timer3"]
]

game_models = [
          "lol/goldnumber-fraction",
          "lol/goldnumber-int",
          "lol/timer-minutes",
          "apex/squad_count_v2",
          "sot/coin_count-digits_3_4-v2",
          "sot/timer3"
          ]

noscope_models = [
        "noscope_coral",
        "noscope_night",
        "noscope_taipei",
        "noscope_amsterdam",
        "noscope_roundabout"
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=game_models + noscope_models + efficientnet_game_models,
                        help="Model to profile")
    parser.add_argument("--do_fold", action="store_true",
                        help="Whether to fold")
    parser.add_argument("--fold", type=int, default=1,
                        help="Number of images to fold into a single input")
    parser.add_argument("--channels_mult", type=float,
                        help="Amount to increase channels")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size to profile at")
    parser.add_argument("--config_dir", type=str, default="config",
                        help="Path to configuration directory")
    parser.add_argument("--depth_mult", type=float, default=1.2)
    parser.add_argument("--width_mult", type=float, default=1.1)
    parser.add_argument("--res_mult", type=float, default=1.15)
    args = parser.parse_args()

    if args.channels_mult is None:
        args.channels_mult = (args.fold ** 0.5)

    if not args.do_fold and (args.fold != 1 or args.channels_mult != 1.):
        raise Exception("Fold and channels_mult must be set to 1 for mode "
                        "'none'")

    if "efficientnet" in args.model and "game" in args.model:
        model_cfg = '_'.join(args.model.split('_')[1:])
        model_conf_dir = os.path.join(args.config_dir, model_cfg)
        desc_file = os.path.join(model_conf_dir, "model_description.json")
        with open(desc_file) as infile:
            cfgs = json.load(infile)

        cfgs['num-classes'] = len(cfgs["ClassNames"])
        cfgs['chan-in'] = 3

        model = efficientnet_game.ConvN(cfgs, fold=args.fold,
                             depth_mult=args.depth_mult,
                             width_mult=args.width_mult,
                             res_mult=args.res_mult,
                             do_fold=args.do_fold)
        in_h = int(cfgs['out-h'] / args.res_mult) if args.do_fold else cfgs['out-h']
        in_w = int(cfgs['out-w'] / args.res_mult) if args.do_fold else cfgs['out-w']
        in_chan = 3

    elif "noscope" in args.model:
        model_name = args.model.split('_')[-1]
        model = noscope.noscope_cnn(model_name,
                                    do_fold=args.do_fold,
                                    fold=args.fold,
                                    internal_multiplier=args.channels_mult)
        in_h = 50
        in_w = 50
        in_chan = 3
        if args.do_fold:
            in_chan *= args.fold

    else:
        model_conf_dir = os.path.join(args.config_dir, args.model)
        desc_file = os.path.join(model_conf_dir, "model_description.json")
        with open(desc_file) as infile:
            cfgs = json.load(infile)

        cfgs['num-classes'] = len(cfgs["ClassNames"])
        cfgs['chan-in'] = 3

        model = game.ConvN(cfgs, fold=args.fold,
                             channel_mult=args.channels_mult,
                             do_fold=args.do_fold)
        in_h = cfgs['out-h']
        in_w = cfgs['out-w']

        in_chan = 3
        if args.do_fold:
            in_chan *= args.fold

    model.eval()
    in_data = torch.rand(args.batch_size, in_chan, in_h, in_w)
    print(in_data.size())
    macs, params = profile(model, inputs=(in_data,))
    print(int(2 * macs))
