# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: cberube & gdupuis
# @Date:   07-10-2019
# @Email:  charles.lafreniere-berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2020-12-21 21:12:69


import numpy as np
import torch
import torch.nn.functional as F


def pixel_acc(output, target, th=0.5):
    ''' Pixel-wise accuracy metric '''
    pred = torch.sigmoid(output.squeeze(1)) >= th
    N_correct = (pred.float() == target.squeeze(1)).sum().item()
    N_correct = N_correct/(target.shape[2]*target.shape[3])
    return N_correct


def iou_acc(output, target, smooth=1.0e-6, th=0.5):
    ''' Intersection over union metric '''
    pred = F.softmax(output, dim=1) >= th
    class_iou = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        t = (target == i).int()
        p = pred[:, i].int()
        intersection = (p & t).sum((1, 2)).float()
        union = (p | t).sum((1, 2)).float()
        iou = (intersection + smooth)/(union + smooth)
        class_iou[i] = iou.sum()
    return class_iou


def iou_acc_numpy(pred, target, class_number, smooth=1.0e-6, robust_iou=False):
    ''' Intersection over union metric '''
    if robust_iou:
        idx = np.where((target == class_number).any((-2, -1)))[0]
    else:
        idx = slice(0, -1)
    pred = pred[:, class_number]
    target = (target == class_number)
    intersection = (pred[idx] & target[idx]).sum((1, 2))
    union = (pred[idx] | target[idx]).sum((1, 2))
    iou = (intersection + smooth)/(union + smooth)
    return iou.mean(), iou.std()


def get_iou(outputs: np.array, labels: np.array, smooth=1e-6):
    # returns the Intersection over Union score for a batch
    # outputs shape: (batch_size, channels, height, width)
    # labels shape: (batch_size, channels, height, width)
    # outputs = outputs.squeeze(1)
    # labels = labels.squeeze(1)
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou


def iou_not_empty(model, test_loader, class_number, threshold=0.5):
    """Returns IoU for non-empty image tiles only
    Args:
        test_loader: A pytorch test set DataLoader
        threshold (float): probability value over which predictions are True
            (default: 0.5).
    Returns:
        tuple: the average and standard deviation of the IoU over the test set
    """
    device = next(model.parameters()).device
    iou_list = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, classes = batch['X'], batch['y']
            classes = classes.numpy().astype('uint8')
            idx = np.where((classes == class_number).any((-2, -1)))[0]
            outputs = model(inputs.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = probs >= threshold
            iou = get_iou(preds[idx, class_number], classes[idx]).sum()
            iou_list.append(iou/len(idx))
    iou_avg = np.mean(iou_list)
    iou_std = np.std(iou_list)
    print(f'Test IoU: {iou_avg:.3f} +/- {iou_std:.3f}')
    return iou_avg, iou_std
