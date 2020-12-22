#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   04-11-2019
# @Email:  charles@goldspot.ca
# @Last modified by:   charles
# @Last modified time: 2020-05-26T10:46:40-04:00


import os
import getpass
import argparse

from pipeline.train import make_target_raster
from pipeline.test import make_testing_tiles
from pipeline.test import test_tiles


def main():
    n_classes = len(args.src_vectors) + 1
    model_id = os.path.splitext(os.path.basename(args.model_path))[0]
    temp_dir = os.path.join('./tmp/', args.username, model_id)
    temp_path = os.path.join(temp_dir, 'target.tif')

    make_target_raster(src_paths=args.src_vectors,
                       template_path=args.src_raster,
                       dst_path=temp_path)

    make_testing_tiles(input_path=args.src_raster,
                       target_path=temp_path,
                       dst_dir=temp_dir,
                       tile_size=args.tile_size,
                       tile_step=args.tile_step)

    # Run the predictions on the input tiles
    test_tiles(src_dir=temp_dir,
               batch_size=args.batch_size,
               model_path=args.model_path,
               n_classes=n_classes,
               threshold=args.threshold,
               robust_iou=args.robust_iou,
               input_channels=args.input_channels)

    os.system('rm -rf {}'.format(temp_dir))


if __name__ == '__main__':
    # Define arguments for CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='path to the pretrained model')
    parser.add_argument('raster', type=str,
                        help='path to the input test raster')
    parser.add_argument('vectors', type=str, nargs='+',
                        help='paths to the target test vectors')
    # Optional arguments
    parser.add_argument('--username', type=str, default=getpass.getuser(),
                        help='name of current user')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='probability threshold to compute IoU')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='number of tiles in each testing batch')
    parser.add_argument('--tile-size', type=int, default=128,
                        help='pixel size of the tiles')
    parser.add_argument('--tile-step', type=int, default=None,
                        help='pixel step between two consecutive tiles')
    parser.add_argument('--input-channels', type=int, nargs='+', default=None,
                        help='indices of the image channels to test on')
    parser.add_argument('--robust-iou', action='store_true',
                        help=('toggle to compute IoU only on tiles '
                              'where class is present'))

    args = parser.parse_args()
    main()
