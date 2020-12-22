# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: charles
# @Date:   29-10-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 21:12:48


import os
import ast

import h5py
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import DataLoader

from detector.utils import OrthoData
from detector.utils import predict
from detector.utils import parse_dirs
from detector.unet import UNet
from detector.metrics import iou_acc_numpy
from detector.preprocessing import make_nd_windows


def save_model_scores(model_id, test_iou_avg, test_iou_std):
    with open('model_descriptions.txt', 'r') as file:
        all_lines = file.readlines()
        for num, line in enumerate(all_lines):
            if model_id in line:
                model_line = num

        dict_line = ast.literal_eval(all_lines[model_line])
        params = dict_line[model_id]
        params['test_iou_avg'] = test_iou_avg
        params['test_iou_std'] = test_iou_std
        sorted_params = {model_id: dict(sorted(params.items()))}
        all_lines[model_line] = str(sorted_params) + '\n'

    with open('model_descriptions.txt', 'w') as file:
        file.writelines(all_lines)


def make_testing_tiles(input_path, target_path, dst_dir, tile_size=128,
                       tile_step=None):

    print('Making testing tiles')
    if tile_step is None:
        tile_step = tile_size

    os.makedirs(dst_dir, exist_ok=True)

    with rio.open(input_path) as src:
        input_data = src.read()
    print('Loaded input file: {}'.format(input_path))

    with rio.open(target_path) as src:
        target_data = src.read()
    print('Loaded target file: {}'.format(target_path))

    input_tiles = make_nd_windows(input_data,
                                  tile_size,
                                  tile_step,
                                  axis=(1, 2))
    target_tiles = make_nd_windows(target_data,
                                   tile_size,
                                   tile_step,
                                   axis=(1, 2))

    if input_tiles.ndim < 5:
        input_tiles = np.expand_dims(input_tiles, 2)
    if target_tiles.ndim < 5:
        target_tiles = np.expand_dims(target_tiles, 2)

    print('Input windows shape:', input_tiles.shape)
    # (n_window_x, n_window_y, n_channels, window_height, window_width)
    print('Output windows shape:', target_tiles.shape)
    # (n_window_x, n_window_y, window_height, window_width)

    mask = (input_tiles == 0).all(axis=2).any(axis=(-1, -2))

    X = (input_tiles[~mask]).astype('uint8')
    y = (target_tiles[~mask]).astype('uint8')

    # Save as HDF
    train_path = os.path.join(dst_dir, 'test_tiles.h5')
    with h5py.File(train_path, 'w') as hf:
        hf.create_dataset('X', data=X)
        hf.create_dataset('y', data=y)

    print()


def test_tiles(src_dir, dst_dir=None, batch_size=64, model_path=None,
               n_classes=3, threshold=0.5, robust_iou=False,
               input_channels=None):

    if not model_path:
        raise RuntimeError('Please pass the path to saved model weights '
                           'using the model_path kwarg')

    src_dir, dst_dir = parse_dirs(src_dir, dst_dir)

    data_path = src_dir + 'test_tiles.h5'

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
    print('Predicting over test set using batch_size={}'.format(batch_size))
    inputs, preds, probs, targets = predict(model, data_loader, th=threshold)

    test_iou_avg = []
    test_iou_std = []
    for i in range(n_classes):
        iou_avg, iou_std = iou_acc_numpy(preds, targets, class_number=i,
                                         robust_iou=robust_iou)
        test_iou_avg.append(iou_avg)
        test_iou_std.append(iou_std)
        print('Class {} IoU: {:.3f} +/- {:.3f}'.format(i, iou_avg, iou_std))
    print()

    model_id = os.path.splitext(os.path.basename(model_path))[0]
    save_model_scores(model_id, test_iou_avg, test_iou_std)
