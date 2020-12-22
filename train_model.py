#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   04-11-2019
# @Email:  charles@goldspot.ca
# @Last modified by:   cberube
# @Last modified time: 27-01-2020


import os
import getpass
import argparse
import datetime

from pipeline.train import make_target_raster
from pipeline.train import make_training_tiles
from pipeline.train import train_on_tiles
from pipeline.train import save_model_parameters
from pipeline.test import make_testing_tiles
from pipeline.test import test_tiles


def main():
    n_classes = len(args.vectors) + 1

    model_id = datetime.datetime.now().replace(microsecond=0)
    model_id = model_id.isoformat().replace(':', '-')
    print('\nUsername: {}'.format(getpass.getuser()))
    print('Model ID: {}'.format(model_id))

    # Make some directories
    temp_dir = os.path.join('./tmp/', args.username, model_id)
    temp_path = os.path.join(temp_dir, 'target.tif')

    # Start the pipeline processing by rasterizing the target shapefiles
    print()
    make_target_raster(src_paths=args.vectors,
                       template_path=args.raster,
                       dst_path=temp_path)

    make_training_tiles(input_path=args.raster,
                        target_path=temp_path,
                        dst_dir=temp_dir,
                        tile_size=args.tile_size,
                        tile_step=args.tile_step,
                        test_size=args.val_size)

    train_on_tiles(data_dir=temp_dir,
                   model_id=model_id,
                   max_epochs=args.max_epochs,
                   learning_rate=args.learning_rate,
                   batch_size=args.batch_size,
                   threshold=args.threshold,
                   n_classes=n_classes,
                   class_weights=args.class_weights,
                   batch_norm=args.batch_norm,
                   augmentation=args.augmentation,
                   input_channels=args.input_channels,
                   scheduler=args.scheduler)

    save_model_parameters(model_id, vars(args))

    if args.test_auto:
        print()
        print('Test flag -t enabled: model will now be tested on South Block')
        raster = 'rasters/South_Block_mosaic_RGBA-NDWI-NDVI-gray_1m.tif'
        vectors = ['vectors/test/South_Block_waterbodies.shp',
                   'vectors/test/South_Block_outcrops.shp']
        model_path = './saved_weights/{}.pt'.format(model_id)

        make_target_raster(src_paths=vectors,
                           template_path=raster,
                           dst_path=temp_path)

        make_testing_tiles(input_path=raster,
                           target_path=temp_path,
                           dst_dir=temp_dir)

        # Run the predictions on the input tiles
        test_tiles(src_dir=temp_dir,
                   batch_size=args.batch_size,
                   model_path=model_path,
                   n_classes=n_classes,
                   threshold=args.threshold,
                   input_channels=args.input_channels)

    os.system('rm -rf {}'.format(temp_dir))

    if args.push_git:
        os.system('git add .')
        os.system("git commit -m 'Update model descriptions'")
        os.system('git push')


if __name__ == '__main__':
    # Define arguments for CLI
    options = {'formatter_class': argparse.ArgumentDefaultsHelpFormatter,
               }
    parser = argparse.ArgumentParser(**options)
    # Mandatory inputs paths
    parser.add_argument('raster', type=str,
                        help='path to the input raster file')
    parser.add_argument('vectors', type=str, nargs='+',
                        help='paths to the target vector files')
    # Optional arguments
    parser.add_argument('-a', '--augmentation', action='store_true',
                        help='toggle image augmentation to training process')
    parser.add_argument('-s', '--scheduler', action='store_true',
                        help='toggle use of learning rate scheduler')
    parser.add_argument('-t', '--test-auto', action='store_true',
                        help='toggle automatic testing on the South block')
    parser.add_argument('-p', '--push-git', action='store_true',
                        help=('toggle automatic git push -- '
                              'ONLY USE IF YOU KNOW WHAT YOU ARE DOING'))
    parser.add_argument('-b', '--batch-norm', action='store_false',
                        help='toggle batch normalization')
    parser.add_argument('--username', type=str, default=getpass.getuser(),
                        help='name of current user')
    # Optional tiling arguments
    parser.add_argument('--tile-size', type=int, default=128,
                        help='pixel size of the tiles')
    parser.add_argument('--tile-step', type=int, default=None,
                        help='pixel step between two consecutive tiles')
    # Optional training arguments
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='fraction size of the validation subset')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='probability threshold to compute IoU')
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='training learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='number of tiles in each training batch')
    parser.add_argument('--class-weights', type=int,  nargs='+', default=None,
                        help='relative weights of the target classes')
    parser.add_argument('--input-channels', type=int, nargs='+', default=None,
                        help='indices of the image channels to train on')

    args = parser.parse_args()
    main()
