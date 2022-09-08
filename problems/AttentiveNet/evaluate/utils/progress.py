# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if logger is None:
            print('\t'.join(entries))
        else:
            logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,), per_sample=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum() #sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if per_sample:
            return res,correct
        else:
            return res

def dissim(vector1, vector2):
    """Dissimalrity scores between 1D vectors -- to be used for entropy difference scoring""" 
    return 1 - torch.dot(vector1, vector2)

def entropy(output):
    """Estimates entropy of classification for each batch"""
    with torch.no_grad():
        output = F.softmax(output) # to avoid NaN
        if len(output.shape) == 1:
            entropy_mtx = - torch.sum(torch.mul(output, torch.log(output)))
        else:
            entropy_mtx = - torch.sum(torch.mul(output, torch.log(output)), 1)
        return entropy_mtx

def accuracy_exits(output, target, topk=(1,), n_exits=1, per_sample=False):
    """Computes the accuracy over the k top predictions for the specified values of k for all the exits"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        res_exits = []
        correct_exits = []

        for i in range(n_exits+1):
            _, pred = output[i].topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_exits.append(correct)
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum() #sum(0, keepdim=True)
                res.append((correct_k.mul_(100.0 / batch_size)))
            
            res_exits.append(res)
        
        if per_sample:
            return res_exits, correct_exits
        else:
            return res_exits

def entropy_exits(outputs, n_exits):
    """Estimates entropy of classification for each batch and for each exit"""
    entropy_exits = []
    with torch.no_grad():
        for i in range(n_exits+1):
            output = outputs[i]
            output = F.softmax(output) # to avoid NaN
            if len(output.shape) == 1:
                entropy_mtx = - torch.sum(torch.mul(output, torch.log(output)))
                entropy_exits.append(entropy_mtx)
            else:
                entropy_mtx = - torch.sum(torch.mul(output, torch.log(output)), 1)
                entropy_exits.append(entropy_mtx)
    return entropy_exits