# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube & gdupuis
# @Date:   17-10-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 21:12:39


import os
import errno
import copy
import time

import numpy as np
import h5py
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .metrics import iou_acc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def write_raster(dst_path, dst_arr, dst_meta, verbose=False):
    """Utility function that writes a N-D array to GeoTiff.

    Depending on the shape of the array passed, will write a raster with
    one or multiple bands. The destination metadata

    Args:
        dst_path (str): The destination path.
        dst_arr (array): A numpy array of the destination data.
            dst_arr.shape = (n_bands, height, width).
        dst_meta (dict): The destination metadata in rasterio metadata format.

    @author: charles
    """
    with rasterio.open(dst_path, 'w', **dst_meta) as dst:
        count = dst.meta['count']
        for i in range(count):
            dst.write_band(i+1, dst_arr[i])
            if verbose and count > 1:
                print(f"--> Writing image band ({i+1}/{count})")


def parse_dirs(src_dir, dst_dir):
    if not dst_dir:
        dst_dir = src_dir
    src_dir = os.path.join(src_dir, '')
    dst_dir = os.path.join(dst_dir, '')
    os.makedirs(dst_dir, exist_ok=True)
    return src_dir, dst_dir


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:"
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def count_parameters(model):
    # Returns the number of parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class ImgAugTransform:
    """Class needed to pass imgaug augmenters to pytorch CustomDataSet
    """

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y):

        x = np.array(x)
        y = np.array(y, dtype='uint8')

        # print(x.shape, y.shape)

        x = np.moveaxis(x, 1, -1)  # reshape x to imgaug
        # y = np.expand_dims(y, -1)
        # print(x.shape, y.shape)

        x_aug, y_aug = self.aug(images=x, segmentation_maps=y)

        # print(x.shape, y.shape)

        x_aug = np.moveaxis(x_aug, -1, 1)
        # y_aug = y_aug.squeeze()

        return (torch.Tensor(x_aug),
                torch.LongTensor(y_aug))


class OrthoData(Dataset):
    def __init__(self, filepath, norm=None, transform=None, channels=None):
        self.filepath = filepath
        self.norm = norm
        self.transform = transform
        if isinstance(channels, int):
            channels = [channels]
        self.channels = channels

        with h5py.File(self.filepath, 'r') as hf:
            if 'X' in list(hf.keys()):
                self.input_shape = hf['X'].shape
            if 'y' in list(hf.keys()):
                self.target_shape = hf['y'].shape
                self.n_classes = np.max(hf['y']) + 1

    def __len__(self):
        with h5py.File(self.filepath, 'r') as hf:
            N = hf['X'].shape[0]
        return N

    def __getitem__(self, i):
        with h5py.File(self.filepath, 'r') as hf:
            X = hf['X'][i]
            if 'y' in list(hf.keys()):
                y = hf['y'][i]
            else:
                y = None
        if y is not None:
            if self.transform:
                X, y = self.transform(X, y)
            y = torch.LongTensor(y)
        X = torch.Tensor(X)
        if self.norm:
            X = self.norm(X)
        if self.channels:
            X = X[self.channels, :, :]
        if y is None:
            return{'X': X}
        else:
            return {'X': X, 'y': y.squeeze()}


def train_model(model, dataloaders, epochs,
                lossfcn, optimizer, scheduler=None,
                return_history=True, transform=None):
    '''
    Train model with cross-validation.
    dataloaders is dictionary-like with 'train' and 'val' dataloaders
    '''
    history = {'train': {'loss': [], 'accuracy': []},
               'val': {'loss': [], 'accuracy': []}
               }
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0e10  # best_acc = 0.0

    for j in range(epochs):
        print('Epoch {}/{}'.format(j+1, epochs))
        print('-----------')

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = np.array([0.0, 0.0, 0.0])

        for batch in dataloaders['train']:

            if transform:
                batch['X'], batch['y'] = transform(batch['X'], batch['y'])

            batch['X'] = batch['X'].to(device)
            batch['y'] = batch['y'].to(device)
            optimizer.zero_grad()
            outputs = model(batch['X'])
            loss = lossfcn(outputs, batch['y'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*batch['X'].size(0)

            # for i in range(3):
            running_corrects += iou_acc(outputs, batch['y'])

        epoch_loss = running_loss/len(dataloaders['train'].dataset)
        epoch_acc = running_corrects/len(dataloaders['train'].dataset)
        if return_history:
            history['train']['loss'].append(epoch_loss)
            history['train']['accuracy'].append(epoch_acc)

        print('Training Loss: {:.4f} Acc: {}'.format(epoch_loss, epoch_acc))

        #Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = np.array([0.0, 0.0, 0.0])

        for batch in dataloaders['val']:
            batch['X'] = batch['X'].to(device)
            batch['y'] = batch['y'].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(batch['X'])
                loss = lossfcn(outputs, batch['y'])
            running_loss += loss.item()*batch['X'].size(0)
            running_corrects += iou_acc(outputs, batch['y'])

        epoch_loss = running_loss/len(dataloaders['val'].dataset)
        epoch_acc = running_corrects/len(dataloaders['val'].dataset)

        if return_history:
            history['val']['loss'].append(epoch_loss)
            history['val']['accuracy'].append(epoch_acc)

        if scheduler:
            scheduler.step(epoch=(j+1), metrics=epoch_loss)

        print('Validation Loss: {:.4f} Acc: {}'.format(epoch_loss, epoch_acc))
        print('Learning rate: {:.3e}'.format(optimizer.param_groups[0]['lr']))
        print()

        if epoch_loss < best_loss: #epoch_acc > best_acc:
            best_loss = epoch_loss #best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    history['val']['accuracy'] = np.array(history['val']['accuracy'])
    history['train']['accuracy'] = np.array(history['train']['accuracy'])

    elapsed = time.time() - start_time
    print('Finished training. Total time: {:.0f}m {:.0f}s'.format(elapsed//60, elapsed%60))
    #print('Best validation acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    if return_history:
        return model, history
    else:
        return model


def find_lr(model, dataloader, lossfcn, optimizer,
            init_value=1.0e-8, final_value=10.0, beta=0.98):
    num = len(dataloader) - 1
    mult = (final_value)/(init_value)**(1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses, log_lrs = [], []
    for batch in dataloader:
        batch_num += 1
        # Get loss for mini-batch
        optimizer.zero_grad()
        batch['X'] = batch['X'].to(device)
        batch['y'] = batch['y'].to(device)
        outputs = model(batch['X'])
        loss = lossfcn(outputs, batch['y'])
        # Compute smoothed loss
        avg_loss = beta*avg_loss + (1-beta)*loss.item()
        smoothed_loss = avg_loss/(1-beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4*best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        # Do optimizaton step
        loss.backward()
        optimizer.step()
        # Update the lr
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


def predict(model, data, th=0.5):
    model.eval()
    with torch.no_grad():
        if isinstance(data, DataLoader):

            dataiter = iter(copy.deepcopy(data)).next()['X']
            n_ex = len(data.dataset)

            n_classes = model.n_classes
            input_shape = (n_ex, *dataiter.shape[-3:])
            target_shape = (n_ex, *dataiter.shape[-2:])
            probs_shape = (n_ex, n_classes, *dataiter.shape[-2:])

            inputs = np.zeros(input_shape, dtype='uint8')
            targets = np.zeros(target_shape, dtype='uint8')
            probs = np.zeros(probs_shape, dtype='float32')
            preds = probs.astype('uint8')

            for i, batch in enumerate(data):
                idx = i*data.batch_size
                b_size = len(batch['X'])
                batch['X'] = batch['X'].to(device)
                outputs = model(batch['X'])
                prob = F.softmax(outputs, dim=1)
                pred = prob >= th
                inputs[idx:idx+b_size] = batch['X'].cpu().numpy()
                preds[idx:idx+b_size] = pred.cpu().numpy()
                probs[idx:idx+b_size] = prob.cpu().numpy()
                if 'y' in list(batch.keys()):
                    targets[idx:idx+b_size] = batch['y'].cpu().numpy()

        else:
            inputs = data['X'].to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs >= th
            if 'y' in list(batch.keys()):
                targets = data['y'].cpu().numpy()
            inputs = inputs.cpu().numpy()

    return inputs, preds, probs, targets
