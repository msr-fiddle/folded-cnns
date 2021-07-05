import argparse
from functools import partial
import os
import pandas as pd
from shutil import copyfile
from tqdm import tqdm


def get_frame_vals(key, infile):
    df = pd.read_csv(args.infile)
    total_num_frames = max(df["frame"]) + 1
    frames_with_key = df.loc[df["object_name"] == key]["frame"].unique()
    frames = [False] * total_num_frames
    for f in frames_with_key:
        frames[f] = True
    return frames


config = {
    "coral": {
            "frame_fn": partial(get_frame_vals, "person"),
            "fraction_split": 1/8,
            "start_split": 5
        },
    "night": {
            "frame_fn": partial(get_frame_vals, "car"),
            "fraction_split": 1/8,
            "start_split": 1
        },
    "roundabout": {
            "frame_fn": partial(get_frame_vals, "car"),
            "fraction_split": 1/8,
            "start_split": 1
        },
    "taipei": {
            "frame_fn": partial(get_frame_vals, "bus"),
            "fraction_split": 1/16,
            "start_split": 0
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, choices=config.keys(),
                        help="NoScope configuration to use")
    parser.add_argument("outdir", type=str,
                        help="Directory to write dataset to")
    parser.add_argument("--infile", type=str,
                        default="coral-reef-long.csv",
                        help="Input CSV file from NoScope")
    parser.add_argument("--indir", type=str,
                        default="noscope-frames-all",
                        help="Directory containing all NoScope frames")
    parser.add_argument("--print_stats", action="store_true",
                        help="Whether to print fraction of frames with object")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise Exception("Input CSV file '{}' does not exist".format(args.indir))

    if not os.path.isdir(args.indir):
        raise Exception("Input directory '{}' does not exist".format(args.indir))

    cfg = config[args.config]
    fraction_split = cfg["fraction_split"]
    start_split = cfg["start_split"]

    num_splits = int(1 / fraction_split)
    if start_split > (num_splits - 3):
        raise Exception("Start split of {} is too high for set with {} splits".format(
            start_split, num_splits))

    frames = cfg["frame_fn"](args.infile)
    total_num_frames = len(frames)
    frames_per_split = int(total_num_frames * fraction_split)

    train_start = start_split * frames_per_split
    val_start = (start_split + 1) * frames_per_split
    test_start = (start_split + 2) * frames_per_split
    test_end = (start_split + 3) * frames_per_split
    train_frame_nos = (train_start, val_start)
    val_frames_nos = (val_start, test_start)
    test_frames_nos = (test_start, test_end)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    dirs = [("train", train_frame_nos), ("val", val_frames_nos), ("test", test_frames_nos)]

    if args.print_stats:
        for name, frames_start_end in dirs:
            start, end = frames_start_end
            num_true = len([f for f in frames[start:end] if f])
            print(name, num_true / (end - start))

    for name, frames_start_end in dirs:
        set_outdir = os.path.join(args.outdir, name)
        if not os.path.isdir(set_outdir):
            os.makedirs(set_outdir)

        # Make the class subdirectories. 0 represents frames without the object
        # of interest. 1 represents those with the objeect of interest.
        dir0 = os.path.join(set_outdir, "0")
        dir1 = os.path.join(set_outdir, "1")
        if not os.path.isdir(dir0):
            os.makedirs(dir0)
        if not os.path.isdir(dir1):
            os.makedirs(dir1)

        # Iterate through each frame in the set, determine whether its class,
        # determine its location in the input dataset, and copy it to the
        # corresponding location in the output directory.
        for frame_no in tqdm(range(frames_start_end[0], frames_start_end[1])):
            if frames[frame_no]:
                class_outdir = dir1
            else:
                class_outdir = dir0

            frame_in_path = os.path.join(args.indir, "{}.png".format(frame_no))
            frame_out_path = os.path.join(class_outdir, "{}.png".format(frame_no))
            copyfile(frame_in_path, frame_out_path)
