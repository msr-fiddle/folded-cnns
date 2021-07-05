""" Script for computing FLOPs/sec from throughput """
import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "tput", type=float,
            help="Throughput in images/sec")
    parser.add_argument(
            "model", type=str,
            help="Model to compute FLOPs/sec for")
    parser.add_argument(
            "mode", type=str, choices=["og", "fold2", "fold3", "fold4"],
            help="Mode in which the model was run")
    parser.add_argument(
            "--config_dir", type=str, default="config",
            help="Path to configuration directory")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, "model_map.json"), 'r') as infile:
        model_map = json.load(infile)

    if "efficientnet" not in args.model:
        # Handle the case where `model` was passed in as 'game_<model_name>'
        args.model = args.model.split('game_')[-1]

    # The FLOP count in `model_map` is represented as FLOPs for batch size 1.
    # For the folded-variants, this is really the FLOPs used for `fold` images.
    # We divide by the number of images represented in the FLOP count so as
    # to truly get FLOPs/sec.
    if args.mode == "og" or "efficientnet" in args.model:
        div = 1
    elif args.mode == "fold2":
        div = 2
    elif args.mode == "fold3":
        div = 3
    elif args.mode == "fold4":
        div = 4

    flop_count = model_map[args.model][args.mode + "_ops"] / div
    print("{:.2f}".format(flop_count * args.tput))
