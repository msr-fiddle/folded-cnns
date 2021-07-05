"""
Script for saving video frames to png files

  Adapted from NoScope project:
    https://github.com/stanford-futuredata/noscope/blob/ffc53d415a6075258a766c01621abcc65ff71200/noscope/VideoUtils.py
"""

import cv2
import numpy as np
from math import ceil
import os

def VideoIterator(video_fname, scale=None, interval=1, start=0):
    cap = cv2.VideoCapture(video_fname)
    # Seeks to the Nth frame. The next read is the N+1th frame
    # In OpenCV 2.4, this is cv2.cv.CAP_PROP_POS_FRAMES (I think)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    frame = 0
    frame_ind = -1
    if scale is not None:
        try:
            len(scale)
            resol = scale
            scale = None
        except:
            resol = None
    while frame is not None:
        frame_ind += 1
        _, frame = cap.read()
        if frame_ind % interval != 0:
            continue
        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        elif resol is not None:
            frame = cv2.resize(frame, resol, interpolation=cv2.INTER_NEAREST)
        yield frame_ind, frame


def get_all_frames(video_fname, outdir, scale=None, interval=1, start=0,
                   print_interval=1000, end=None):
    i = 0
    for _, frame in VideoIterator(video_fname, scale=scale, interval=interval, start=start):
        cv2.imwrite(os.path.join(outdir, str(i) + '.png'), frame)
        i += 1

        if i % print_interval == 0:
            print(i)

        if end is not None and i > end:
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="MP4 file containing NoScope video")
    parser.add_argument("outdir", type=str,
                        help="Directory to which to save video frames")
    parser.add_argument("--start", type=int, default=0,
                        help="Frame to start saving from")
    parser.add_argument("--end", type=int,
                        help="Last frame to save from")
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    get_all_frames(args.infile, outdir=args.outdir, scale=(50, 50), start=args.start, end=args.end)
