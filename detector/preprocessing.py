# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gdupuis & cberube
# @Date:   07-10-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 21:12:42


import h5py
import numpy as np
import torch
try:
    import imgaug.augmenters as iaa
except ImportError:
    print("imgaug not installed. Do not use the -a --augmentation flags.")


def make_nd_windows(a, window, steps=None, axis=None, outlist=False):
    """
    Create a windowed view over `n`-dimensional input that uses an
    `m`-dimensional window, with `m <= n`

    Parameters
    -------------
    a : Array-like
        The array to create the view on

    window : tuple or int
        If int, the size of the window in `axis`, or in all dimensions if
        `axis == None`

        If tuple, the shape of the desired window.  `window.size` must be:
            equal to `len(axis)` if `axis != None`, else
            equal to `len(a.shape)`, or
            1

    steps : tuple, int or None
        The offset between consecutive windows in desired dimension
        If None, offset is one in all dimensions
        If int, the offset for all windows over `axis`
        If tuple, the steps along each `axis`.
            `len(steps)` must me equal to `len(axis)`

    axis : tuple, int or None
        The axes over which to apply the window
        If None, apply over all dimensions
        if tuple or int, the dimensions over which to apply the window

    outlist : boolean
        If output should be as list of windows.
        If False, it will be an array with
            `a.nidim + 1 <= a_view.ndim <= a.ndim *2`.
        If True, output is a list of arrays with `a_view[0].ndim = a.ndim`
            Warning: this is a memory-intensive copy and not a view

    Returns
    -------

    a_view : ndarray
        A windowed view on the input array `a`, or copied list of windows

    """
    ashp = np.array(a.shape)

    if axis is not None:
        axs = np.array(axis, ndmin=1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin=1)
    assert (window.size == axs.size) | (window.size == 1), (
           "Window dims and axes don't match")
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), (
           "Window is bigger than input array in axes")

    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin=1)
        assert np.all(steps > 0), (
               "Only positive steps allowed")
        assert (steps.size == axs.size) | (steps.size == 1), (
               "Steps and axes don't match")
        stp[axs] = steps

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    as_strided = np.lib.stride_tricks.as_strided
    a_view = np.squeeze(as_strided(a,
                                   shape=shape,
                                   strides=strides))
    if outlist:
        return list(a_view.reshape((-1,) + tuple(wshp)))
    else:
        return a_view


def norm_params(filepath):
    '''
    Get mean and standard deviation of each channel over
    entire dataset, for standard scaling
    '''
    with h5py.File(filepath, 'r') as hf:
        X = hf['X'][:]
    X = X.transpose((0, 2, 3, 1))
    X = X.reshape(-1, X.shape[3])
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    return list(mean), list(stdev)


class minmax_scaler():
    '''
    MinMax scale over entire dataset.
    '''

    def __init__(self, filepath, ft_range=(0.0, 1.0)):
        ''' filepath: name of hdf5 file containing dataset on which to fit '''
        self.ft_range = ft_range

        with h5py.File(filepath, 'r') as hf:
            X = hf['X'][:]  # X dim is (N, c, h, w)
        X = X.transpose((0, 2, 3, 1))
        X = X.reshape(-1, X.shape[3])
        self.data_min = torch.tensor(X.min(axis=0))
        self.data_max = torch.tensor(X.max(axis=0))
        self.scale = ((self.ft_range[1] - self.ft_range[0])
                      / (self.data_max - self.data_min))

    def __call__(self, X):
        ''' X: tensor with dim (c, h, w) '''
        X = self.scale[:, None, None]*(X - self.data_min[:, None, None])
        return X + self.ft_range[0]


class img_scaler():
    '''
    MinMax scaler for 3d array (H x W x C).
    '''

    def __init__(self, ft_range=(0.0, 1.0)):
        self.ft_range = ft_range

    def fit(self, X):
        self.data_min = X.reshape(-1, X.shape[-1]).min(axis=0)
        self.data_max = X.reshape(-1, X.shape[-1]).max(axis=0)
        self.scale = ((self.ft_range[1] - self.ft_range[0])
                      / (self.data_max - self.data_min))

    def transform(self, X):
        h, w, c = X.shape
        X = X.reshape(h*w, c)
        X = self.scale*(X - self.data_min) + self.ft_range[0]
        X = X.reshape(h, w, c)
        return X

    def inverse_transform(self, X):
        h, w, c = X.shape
        X = X.reshape(h*w, c)
        X = (X - self.ft_range[0])/self.scale + self.data_min
        X = X.reshape(h, w, c)
        return X


class img_stdscaler():
    '''
    Standard scaler for 3d array (H x W x C).
    '''

    def __init__(self, means=None, stdevs=None):
        self.means = means
        self.stdevs = stdevs
        self.fit_local = self.means is None or self.stdevs is None

    def fit(self, X):
        if self.fit_local:
            self.means = X.reshape(-1, X.shape[-1]).mean(axis=0)
            self.stdevs = X.reshape(-1, X.shape[-1]).std(axis=0)

    def transform(self, X):
        h, w, c = X.shape
        X = X.reshape(h*w, c)
        X = (X - self.means)/self.stdevs
        X = X.reshape(h, w, c)
        return X

    def inverse_transform(self, X):
        h, w, c = X.shape
        X = X.reshape(h*w, c)
        X = self.stdevs*X + self.means
        X = X.reshape(h, w, c)
        return X


def img_resize(X, h, w):
    resize = iaa.Resize(size={'height': h, 'width': w})
    X = resize.augment_image(X)
    X = np.clip(X, 0., 1.)
    return X
