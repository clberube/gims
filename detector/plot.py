# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube
# @Date:   18-09-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 21:12:84


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator


def show_image(example, outfile=None):
    X = np.array(example['X']).transpose((1, 2, 0))
    y = np.array(example['y']).transpose((1, 2, 0))
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
    ax[0, 0].imshow(X[:, :, 0])
    ax[0, 0].set_title('Channel 1')
    ax[0, 1].imshow(X[:, :, 1])
    ax[0, 1].set_title('Channel 2')
    ax[1, 0].imshow(X[:, :, 2])
    ax[1, 0].set_title('Channel 3')
    ax[1, 1].imshow(y[:, :, 0])
    ax[1, 1].set_title('Target Mask')
    for axes in ax.flatten():
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches='tight', dpi=300)


def plot_lr(log_lrs, losses, outfile=None):
    plt.figure()
    plt.plot(log_lrs[2:-1], losses[2:-1], color='deeppink')
    plt.xlabel(r'$\log_{10}$' + '(lr)')
    plt.ylabel('Loss')
    if outfile:
        plt.savefig(outfile, dpi=300)


def learning_curve(history, name, outfile=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.plot(history['train'][name], color='C0', label='Training')
    plt.plot(history['val'][name], color='C1', label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()


def compare_learning_curve(history, name, outfile=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    # plt.plot(history['train'][name], color='indigo', label='Training')
    plt.plot(100*history['train'][name][:, 0], color='C2', label='Forest')
    plt.plot(100*history['train'][name][:, 1], color='C0', label='Waterbody')
    plt.plot(100*history['train'][name][:, 2], color='C3', label='Outcrop')
    # plt.plot(history['train'][name][:, 0], color='C2', alpha=0.3)
    # plt.plot(history['train'][name][:, 1], color='C0', alpha=0.3)
    # plt.plot(history['train'][name][:, 2], color='C3', alpha=0.3)
    plt.ylabel('IoU (%)')
    plt.xlabel('Epoch')
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()


def plot_predictions(preds, targets, outfile=None):
    fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 8))
    a = ax.flatten()
    for i in range(8):
        P = np.array(preds[i].squeeze().cpu())
        mask = np.array(targets[i].squeeze().cpu())
        a[2*i].imshow(P, cmap='gnuplot', vmin=0, vmax=1)
        a[2*i+1].imshow(mask, cmap='bone')
    for axes in a:
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches='tight', dpi=300)


def plot_learning_curves(train_losses, valid_losses=None, save_path=None):
    # Plot learning curves
    fig, ax = plt.subplots()
    # epoch_ticks = np.arange(1, max_epochs+1, (max_epochs)/10)
    # epoch_range = range(1, max_epochs+1)
    plt.plot(train_losses, label='Training loss')
    if valid_losses:
        plt.plot(valid_losses, label='Validation loss')
    # plt.plot(epoch_range, epoch_valid_iou, label='Validation IoU')
    # plt.ylabel('Loss')
    plt.xlabel('Number of epochs')
    # plt.xticks(epoch_ticks)
    # ax.set_yticklabels(epochs_range)
    plt.legend()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_results(inputs, classes, prob, n_plot=4, save_path=None):

    condition_1 = (classes == 1).any(axis=(-3, -2, -1))
    condition_2 = (classes == 2).any(axis=(-3, -2, -1))
    condition = condition_1 & condition_2
    non_empty_idx = np.where(condition)[0]
    choice = np.random.choice(len(non_empty_idx), n_plot, replace=False)
    example_idx = non_empty_idx[choice]

    fig, ax = plt.subplots(n_plot, 4, figsize=(8, n_plot*2))

    if inputs.shape[1] >= 3:
        channels_to_plot = slice(3)
    else:
        channels_to_plot = 0

    for row, idx in enumerate(example_idx):

        input_map = np.moveaxis(inputs[idx], 0, -1)
        ca = ax[row, 0]
        im = ca.imshow(input_map[:, :, channels_to_plot])
        divider = make_axes_locatable(ca)

        label_map = classes[idx, 0]
        ca = ax[row, 1]
        im = ca.imshow(label_map, cmap='viridis', vmin=classes.min(),
                       vmax=classes.max())
        divider = make_axes_locatable(ca)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ticks=range(classes.max()+1))

        prob_map = prob[idx, 1]
        ca = ax[row, 2]
        im = ca.imshow(prob_map, cmap='bone', vmin=0, vmax=1)
        divider = make_axes_locatable(ca)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ticks=[0, 1])

        prob_map = prob[idx, 2]
        ca = ax[row, 3]
        im = ca.imshow(prob_map, cmap='bone', vmin=0, vmax=1)
        divider = make_axes_locatable(ca)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ticks=[0, 1])

    ax[0, 0].set_title('Input')
    ax[0, 1].set_title('Label mask')
    ax[0, 2].set_title('Waterbody prob.')
    ax[0, 3].set_title('Outcrop prob.')

    for a in ax.flat:
        a.set_aspect('equal')
        a.set_yticklabels([])
        a.set_xticklabels([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
