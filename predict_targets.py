#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   04-11-2019
# @Email:  charles@goldspot.ca
# @Last modified by:   charles
# @Last modified time: 2020-05-26T10:47:14-04:00


import os
import argparse
import getpass

from pipeline.predict import make_prediction_tiles
from pipeline.predict import predict_tiles
from pipeline.predict import mosaic_prediction_tiles
from pipeline.predict import polygonize_prediction_mosaic


def main():
    model_id = os.path.splitext(os.path.basename(args.model))[0]
    temp_dir = os.path.join('./tmp/', args.username, model_id)
    dst_dir = './{}_exports/{}'.format(args.username, model_id)
    dst_dir = os.path.join(dst_dir, args.subfolder)

    # Start the pipeline processing by making the input tiles
    make_prediction_tiles(src_path=args.raster,
                          dst_dir=temp_dir,
                          tile_size=args.tile_size,
                          tile_step=args.tile_step)

    # Run the predictions on the input tiles
    predict_tiles(src_dir=temp_dir,
                  model_path=args.model,
                  batch_size=args.batch_size,
                  threshold=args.threshold,
                  input_channels=args.input_channels)

    if args.dst_fname is None:
        args.dst_fname = os.path.splitext(os.path.basename(args.raster))[0]
    # Rebuild the mosaic using the prediction tiles
    mosaic_prediction_tiles(src_dir=temp_dir,
                            dst_dir=dst_dir,
                            tile_size=args.tile_size,
                            dst_fname=args.dst_fname,
                            qgis_vis=args.qgis_vis)

    # Polygonize the prediction mosaic with a threshold
    if args.polygonize:
        polygonize_prediction_mosaic(src_dir=dst_dir,
                                     dst_dir=dst_dir,
                                     threshold=args.threshold,
                                     dst_fname=args.dst_fname)

    os.system('rm -rf {}'.format(temp_dir))


if __name__ == '__main__':
    # Mandatory arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='path to the pretrained model')
    parser.add_argument('raster', type=str,
                        help='path to the input test raster')
    # Optional arguments
    parser.add_argument('-p', '--polygonize', action='store_true',
                        help='polygonizes the rasters')
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
    parser.add_argument('--subfolder', type=str, default='',
                        help='subfolder where to save under username_exports')
    parser.add_argument('--dst-fname', type=str, default=None,
                        help='filename for the export files')
    parser.add_argument('--qgis-vis', type=bool, default=True,
                        help='re-organize output channels for QGIS')
    args = parser.parse_args()
    main()
