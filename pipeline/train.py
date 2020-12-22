# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   25-10-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 21:12:00


import os
from pprint import pprint

import h5py
import numpy as np
import rasterio as rio
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
from sklearn.model_selection import train_test_split
try:
    import imgaug.augmenters as iaa
except ImportError:
    print("imgaug not found. Do not use the -a --augmentation flags.")

from detector.utils import OrthoData, ImgAugTransform
from detector.utils import count_parameters, train_model, predict
from detector.utils import GradualWarmupScheduler
from detector.unet import UNet
from detector.plot import plot_results, learning_curve, compare_learning_curve
from detector.preprocessing import make_nd_windows

torch.multiprocessing.set_sharing_strategy('file_system')


def make_target_raster(src_paths, template_path, dst_path=None):

    print('Building target raster from input shapefiles')
    # The final target geotiff that will be exported: does not need to exist
    if dst_path is None:
        dst_path = '.rasters/target_raster.tif'
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Read the template geotiff (the data file)
    with rio.open(template_path) as src:
        data = src.read()
        meta = src.meta

    # Make a copy of the template Geotiff (the data) and fill it with 0's
    dst_meta = meta.copy()
    print('Input metadata:')
    pprint(dst_meta)  # note that the template geotiff has 4 bands
    dst_meta.update(count=1, dtype='uint8')  # only want 1 band in the target geotiff
    dst_data = np.zeros(data.shape[-2:], dtype=dst_meta['dtype'])  # copy the template

    # Write the copied array filled of 0's to band 1 of a geotiff file
    with rio.open(dst_path, 'w', **dst_meta) as dst:
        dst.write_band(1, dst_data)

    print('Rasterizing {} input shapefiles'.format(len(src_paths)))
    for i, s in enumerate(src_paths, start=1):
        # Burn the value '1' in band 1 of the template geotiff where
        gdal_command = 'gdal_rasterize -b 1 -burn {} {} {}'
        os.system(gdal_command.format(i, s, dst_path))

    print('Wrote target raster as {}'.format(dst_path))
    print()


def make_training_tiles(input_path, target_path, dst_dir, tile_size=128,
                        tile_step=None, test_size=0.5):

    print('Making training/validation tiles')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=42)

    print('X_train:', X_train.shape, X_train.dtype)
    print('X_test:', X_test.shape, X_test.dtype)
    print('y_train:', y_train.shape, y_train.dtype)
    print('y_test:', y_test.shape, y_test.dtype)

    # Class counts and weights if needed
    class_0 = np.unique((y_train == 0), return_counts=True)[1][1]
    class_1 = np.unique((y_train == 1), return_counts=True)[1][1]
    class_2 = np.unique((y_train == 2), return_counts=True)[1][1]
    class_max = max([class_0, class_1, class_2])
    class_weights = [int(class_max/class_0),
                     int(class_max/class_1),
                     int(class_max/class_2),
                     ]
    print('Recommended training class weights:', class_weights)

    # Save as HDF
    train_path = os.path.join(dst_dir, 'train_tiles.h5')
    with h5py.File(train_path, 'w') as hf:
        hf.create_dataset('X', data=X_train)
        hf.create_dataset('y', data=y_train)

    test_path = os.path.join(dst_dir, 'test_tiles.h5')
    with h5py.File(test_path, 'w') as hf:
        hf.create_dataset('X', data=X_test)
        hf.create_dataset('y', data=y_test)
    print()


def train_on_tiles(data_dir, model_id, max_epochs=25, learning_rate=1e-3,
                   batch_size=32, threshold=0.5, n_classes=None,
                   class_weights=None, batch_norm=True, augmentation=False,
                   input_channels=None, depth=4, wf=4, padding=True,
                   scheduler=None):

    print('Building U-Net model and training')

    exports_dir = './figures/{}/'.format(model_id)
    os.makedirs(exports_dir, exist_ok=True)

    # Define augmentation transforms
    aug = iaa.Sequential([
        # iaa.MultiplyHueAndSaturation((0.5, 1.5)),
        # iaa.AddToHueAndSaturation((-45, 45)),
        # iaa.Grayscale((0.0, 1.0)),
        # iaa.AllChannelsHistogramEqualization(),
        # iaa.GammaContrast((0.0, 1.75), per_channel=True),
        # iaa.LinearContrast((0.0, 1.75), per_channel=True),
        iaa.Crop(px=(0, 32)),  # randomly crop between 0 and 32 pixels
        iaa.Fliplr(0.50),  # horizontally flip 50% of the images
        iaa.Flipud(0.50),  # horizontally flip 50% of the images
        iaa.Rot90([0, 1, 2, 3]),  # apply any of the 90 deg rotations
        # iaa.PerspectiveTransform((0.025, 0.1)),  # randomly scale between 0.025 and 0.1
        # iaa.Affine(scale=(0.5, 1.5), translate_percent=(0, 0.25),  # random affine transforms with symmetric padding
        #            rotate=(0, 360), shear=(0, 360), mode='symmetric'),
    ])

    if class_weights is None:
        class_weights = [1, 1, 1]

    if augmentation:
        tfms = ImgAugTransform(aug)
    else:
        tfms = None

    datasets = {}
    datasets['train'] = OrthoData(os.path.join(data_dir, 'train_tiles.h5'),
                                  transform=None,
                                  channels=input_channels)
    datasets['val'] = OrthoData(os.path.join(data_dir, 'test_tiles.h5'),
                                transform=None,
                                channels=input_channels)

    if input_channels is None:
        n_channels = datasets['train'].input_shape[1]
    else:
        n_channels = len(input_channels)

    if n_classes is None:
        n_classes = datasets['train'].n_classes

    print('Input channels:', input_channels)
    # DataLoader parameters
    dataloader_params = {'batch_size': batch_size,
                         'num_workers': 0,
                         'shuffle': True}

    # Make the DataLoaders
    data_loaders = {name: DataLoader(datasets[name], **dataloader_params)
                    for name in ['train', 'val']}

    # Define model
    print('CUDA:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet_params = {'n_classes': n_classes,
                   'in_channels': n_channels,
                   'depth': depth,
                   'padding': padding,
                   'wf': wf,
                   'up_mode': 'upconv',
                   'batch_norm': batch_norm}

    model = UNet(**unet_params).to(device)
    n_parameters = count_parameters(model)
    print(f'Number of model parameters: {n_parameters}')
    # Define optimizer and loss function
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = torch.Tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(class_weights)

    if scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                               verbose=True,
                                                               patience=2)
        scheduler = GradualWarmupScheduler(optim, multiplier=10,
                                           total_epoch=10,
                                           after_scheduler=scheduler,
                                           )

    # Train
    model, history = train_model(model, data_loaders, max_epochs,
                                 criterion, optim, scheduler=scheduler,
                                 return_history=True,
                                 transform=tfms)

    os.makedirs('./saved_weights/', exist_ok=True)
    model_filename = '{}.pt'.format(model_id)
    torch.save(model.state_dict(), './saved_weights/{}'.format(model_filename))
    print('Saved model as: ./saved_weights/{}'.format(model_filename))

    n_examples = len(datasets['val'])  # //10
    sampler = RandomSampler(datasets['val'], replacement=True,
                            num_samples=n_examples)
    inputs = DataLoader(datasets['val'], sampler=sampler)

    print('Predicting over the validation dataset')
    inputs, preds, probs, targets = predict(model, data_loaders['val'],
                                            th=threshold)

    plot_results(inputs=inputs.astype(np.uint8),
                 classes=np.expand_dims(targets, 1),
                 prob=probs,
                 n_plot=5,
                 save_path=exports_dir+'example_tiles.png')

    learning_curve(history=history,
                   name='loss',
                   outfile=exports_dir+'loss_learning_curves.png')

    compare_learning_curve(history=history,
                           name='accuracy',
                           outfile=exports_dir+'accuracy_learning_curves.png')


def save_model_parameters(model_id, params):
    sorted_params = {model_id: dict(sorted(params.items()))}
    with open('model_descriptions.txt', 'a') as myfile:
        myfile.write(str(sorted_params) + '\n')
