# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   17-10-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 22:12:74


import os
import pickle
from pprint import pprint

import h5py
import numpy as np
import rasterio as rio
import torch
from torch.utils.data import DataLoader

from detector.unet import UNet
from detector.utils import OrthoData, predict
from detector.utils import parse_dirs
from detector.utils import write_raster
from detector.preprocessing import make_nd_windows

torch.multiprocessing.set_sharing_strategy('file_system')


def make_prediction_tiles(src_path, dst_dir, tile_size=128, tile_step=None):

    dst_dir = os.path.join(dst_dir, '')
    os.makedirs(dst_dir, exist_ok=True)

    with rio.open(src_path) as src:
        src_data = src.read()
        src_meta = src.meta

    src_data[:, (src_data == 255).all(axis=0)] = 0

    print('Making prediction tiles')
    if tile_step is None:
        tile_step = tile_size // 4

    input_tiles = make_nd_windows(src_data, tile_size, tile_step, axis=(1, 2))

    if input_tiles.ndim < 5:
        input_tiles = np.expand_dims(input_tiles, 2)
    print('Input tiles:', input_tiles.shape, input_tiles.dtype)

    # Define a mask to remove tiles with missing values
    mask = (input_tiles == 0).all(axis=2).any(axis=(-1, -2))

    X = (input_tiles[~mask]).astype('uint8')
    print('Flattened tiles:', X.shape, X.dtype)

    # Define a dictionary array to store the tile metadata
    pixel_coords_shape = (input_tiles.shape[0], input_tiles.shape[1], 2)
    pixel_coords_array = np.empty(pixel_coords_shape, dtype=int)
    for i in range(input_tiles.shape[0]):
        for j in range(input_tiles.shape[1]):
            pixel_coords_array[i, j] = (i*tile_step, j*tile_step)

    # Apply the same mask used on the tiles to the metadata array to flatten it
    masked_pixel_coords = (pixel_coords_array[~mask])

    print('Saving tiles, pixel coordinates and '
          'metadata in {}'.format(dst_dir))
    with h5py.File(dst_dir + 'X_tiles.h5', 'w') as hf:
        hf.create_dataset('X', data=X)

    # Pickle tile metadata
    with open(dst_dir + 'X_pixel_coords.pk', 'wb') as f:
        pickle.dump(masked_pixel_coords, f, protocol=4)
    with open(dst_dir + 'X_metadata.pk', 'wb') as f:
        pickle.dump(src_meta, f, protocol=4)


def predict_tiles(src_dir, dst_dir=None, batch_size=64, model_path=None,
                  n_classes=3, threshold=0.5, test=False, input_channels=None):

    if not model_path:
        raise RuntimeError('Please pass the path to saved model weights '
                           'using the model_path kwarg')

    src_dir, dst_dir = parse_dirs(src_dir, dst_dir)

    data_path = src_dir + 'X_tiles.h5'
    pred_filename = dst_dir + 'X_predictions.h5'

    print('Getting best model from {} '.format(model_path))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Generate torch DataSet and DataLoader
    dataset = OrthoData(data_path, norm=None, channels=input_channels)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    if input_channels is None:
        n_channels = dataset.input_shape[1]
    else:
        n_channels = len(input_channels)

    if n_classes is None:
        n_classes = dataset.n_classes

    print('Nb channels:', n_channels)
    print('Nb classes:', n_classes)

    # Define model
    model = UNet(in_channels=n_channels, n_classes=n_classes, wf=4, depth=4,
                 padding=True, batch_norm=True, up_mode='upconv')
    model = model.to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path))

    # Predict
    print('Predicting tiles using batch_size={}'.format(batch_size))
    inputs, preds, probs, targets = predict(model, data_loader, th=threshold)

    # Save predictions
    print('Saving predictions as {}'.format(pred_filename))
    with h5py.File(pred_filename, 'w') as hf:
        hf.create_dataset('X_preds', data=probs)


def mosaic_prediction_tiles(src_dir, dst_dir=None, tile_size=128,
                            dst_dtype='float32', dst_nodata=np.nan,
                            dst_fname=None, qgis_vis=True):

    src_dir, dst_dir = parse_dirs(src_dir, dst_dir)

    if dst_fname is None:
        dst_fname = 'X'

    predictions_path = src_dir + 'X_predictions.h5'
    metadata_path = src_dir + 'X_metadata.pk'
    coords_path = src_dir + 'X_pixel_coords.pk'
    mosaic_path = dst_dir + f'{dst_fname}_{dst_dtype}_prediction_mosaic.tif'
    rgb_mosaic_path = dst_dir + f'{dst_fname}_uint8_prediction_mosaic.tif'

    print('Loading predictions from {}'.format(predictions_path))
    with h5py.File(predictions_path, 'r') as hf:
        preds = hf.get('X_preds')
        preds = np.array(preds, dtype=dst_dtype)
    print('Predictions:', preds.shape, preds.dtype)

    print('Loading metadata from {}'.format(metadata_path))
    with open(metadata_path, 'rb') as input_file:
        dst_meta = pickle.load(input_file)
    print('Metadata:')
    pprint(dst_meta)

    print('Loading image coordinates from {}'.format(coords_path))
    with open(coords_path, 'rb') as input_file:
        coords = pickle.load(input_file)
    print('Coordinates shape:', coords.shape)

    sums_shape = (preds.shape[1], dst_meta['height'], dst_meta['width'])
    counts_shape = (dst_meta['height'], dst_meta['width'])
    mosaic_sums = np.zeros(sums_shape, dtype=dst_dtype)
    mosaic_counts = np.zeros(counts_shape, dtype=dst_dtype)

    print('Stacking tiles and reducing overlaps with mean operation')
    for i, (j, k) in enumerate(coords):
        mosaic_sums[:, j:j+tile_size, k:k+tile_size] += preds[i]
        mosaic_counts[j:j+tile_size, k:k+tile_size] += 1

    mosaic_counts[mosaic_counts == 0] = dst_nodata
    mosaic_mean = (mosaic_sums / mosaic_counts).astype(dst_dtype)

    # Convert RGB to GBR (only for visualization purposes in QGIS)
    if qgis_vis:
        mosaic_mean = mosaic_mean[[2, 0, 1], :]

    dst_meta.update(dtype=dst_dtype, count=preds.shape[1], nodata=dst_nodata)
    write_raster(mosaic_path, mosaic_mean, dst_meta)
    print('Saved mosaic as {}'.format(mosaic_path))

    compressed_mosaic_mean = np.zeros(mosaic_mean.shape, dtype='uint8')
    compressed_meta = dst_meta.copy()
    compressed_meta.update(dtype='uint8', nodata=None)
    for n in range(mosaic_mean.shape[0]):
        compressed_mosaic_mean[n] = np.round(255*mosaic_mean[n])
    write_raster(rgb_mosaic_path, compressed_mosaic_mean, compressed_meta)

    print('Saved compressed RGB mosaic as {}'.format(rgb_mosaic_path))


def polygonize_prediction_mosaic(src_dir, dst_dir=None, dst_subdir='',
                                 threshold=0.5, band=0, dst_fname=None):

    src_dir, dst_dir = parse_dirs(src_dir, dst_dir)
    dst_dir = os.path.join(dst_dir, dst_subdir)
    os.makedirs(dst_dir, exist_ok=True)

    if dst_fname is None:
        dst_fname = 'X'

    src_path = src_dir + f'{dst_fname}_prediction_mosaic.tif'
    with rio.open(src_path) as src:
        data = src.read()
        meta = src.meta
    dst_meta = meta.copy()

    dst_dtype = 'uint8'
    dst_nodata = 0

    dst_meta.update(count=1, dtype=dst_dtype, nodata=dst_nodata)

    for i, band in enumerate(data[1:], start=1):
        dst_basename = ('{}_class{}_'
                        'prediction_'
                        'threshold{:.0%}').format(dst_fname, i, threshold)
        dst_path = os.path.join(dst_dir, dst_basename)

        export = band[np.newaxis, :]
        export[np.isnan(export)] = dst_nodata
        export = (export >= threshold).astype(dst_dtype)
        write_raster(dst_path+'.tif', export, dst_meta)

        os.system('gdal_polygonize.py {} {} {}'.format('-q',
                                                       dst_path+'.tif',
                                                       dst_path+'_tmp.shp'))

        os.system('ogr2ogr {} {} -simplify {}'.format(dst_path+'.shp',
                                                      dst_path+'_tmp.shp',
                                                      src.meta['transform'][0]))

        print('Saved thresholded vector as {}'.format(dst_path+'.shp'))

        os.system('rm -rf {} {}'.format(dst_path+'_tmp.*', dst_path+'.tif'))
